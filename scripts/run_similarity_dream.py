# scripts/run_similarity_dream.py

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import csv
import torch
import argparse
from transformers import AutoTokenizer
from utils.similarity import LayerTap, cosine_matrix, cosine_diag

# Import Dream model wrapper from this repo
from dream.model.modeling_dream import DreamModel


def get_decoder_layers_dream(model):
    """
    Try to collect Dream decoder layers in a few robust ways so the script
    works across minor refactors.
    """
    layers = []

    # Preferred: explicit class name
    for _, mod in model.named_modules():
        if type(mod).__name__ in {"DreamDecoderLayer"}:
            layers.append(mod)

    # Fallback: anything that looks like a Dream block
    if not layers:
        for _, mod in model.named_modules():
            cls = type(mod).__name__
            if "Dream" in cls and "Block" in cls:
                layers.append(mod)

    if not layers:
        raise RuntimeError("Could not find Dream decoder layers")
    return layers


def build_tokenizer(local_model_dir: str | None, default_repo: str):
    """
    Load a tokenizer. Prefer the local model dir if present, otherwise load
    from the public HF repo.
    """
    if local_model_dir:
        try:
            return AutoTokenizer.from_pretrained(local_model_dir, trust_remote_code=True)
        except Exception:
            # Fall back to repo tokenizer if local one is absent/incomplete
            pass
    return AutoTokenizer.from_pretrained(default_repo, trust_remote_code=True)


def build_model(device: str, dtype: torch.dtype, local_model_dir: str | None, hf_repo_id: str):
    """
    Build the Dream model. If a local directory is provided, load from there;
    otherwise pull from the HF repo.
    """
    if local_model_dir:
        # e.g. in build_model()
        if not (Path(local_model_dir) / "config.json").exists():
            raise FileNotFoundError(
                f"`{local_model_dir}` does not look like a checkpoint directory (missing config.json). "
                "Point --local-model-dir to a valid Dream checkpoint (HF-style folder)."
            )
        print(f"[Dream] Loading from {local_model_dir}")
        model = DreamModel.from_pretrained(local_model_dir, torch_dtype=dtype).to(device)
    else:
        print(f"[Dream] Loading from HF repo: {hf_repo_id}")
        # For HF loading, keep trust_remote_code if Dream provides custom code
        model = DreamModel.from_pretrained(hf_repo_id, torch_dtype=dtype, trust_remote_code=True).to(device)

    model.eval()
    return model


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float16

    HF_REPO = args.model_id  # default: Dream-org/Dream-v0-Base-7B

    # Build model & tokenizer
    model = build_model(device, dtype, args.local_model_dir or None, HF_REPO)
    tok = build_tokenizer(args.local_model_dir or None, HF_REPO)

    # Find layers and attach taps
    layers = get_decoder_layers_dream(model)
    print(f"[Dream] tapped {len(layers)} layers")
    tap = LayerTap(layers, pool="last")

    prev = None
    across_path = os.path.join(args.out_dir, "across_steps_cosine.csv")

    with open(across_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step"] + [f"layer_{i}" for i in range(len(layers))])

        def on_step(step: int):
            nonlocal prev
            try:
                A = tap.stacked()  # [L, H] pooled activations per layer
                cos = cosine_matrix(A)
                # ensure CPU tensor for saving (Drive can be picky with device tensors)
                if isinstance(cos, torch.Tensor):
                    cos_cpu = cos.detach().cpu()
                else:
                    cos_cpu = torch.as_tensor(cos)

                pt_path = os.path.join(args.out_dir, f"within_step_cosine_step{step}.pt")
                torch.save(cos_cpu, pt_path)

                # across-steps diagonal
                if prev is not None:
                    row = [step] + cosine_diag(prev, A).tolist()
                    w.writerow(row)
                    # force CSV to hit disk each step
                    f.flush()
                    os.fsync(f.fileno())

                prev = A
                tap.clear()

                # quick heartbeat so you see it’s working
                print(f"[Dream] step {step:02d}: saved {os.path.basename(pt_path)} (shape={tuple(cos_cpu.shape)})")

            except Exception as e:
                print(f"[Dream] on_step error at step {step}: {e}")
                # still try to advance state so subsequent steps keep going
                try:
                    prev = A
                    tap.clear()
                except Exception:
                    pass

        inputs = tok(args.prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            # Call the diffusion generation entrypoint that we instrumented earlier
            _ = model.diffusion_generate(
                inputs=inputs["input_ids"],
                step_callback=on_step,         # ✦ hook (fires once per diffusion step)
                steps=args.steps,              # diffusion steps
                block_length=args.block_size,  # block size to align with logging
                temperature=0.0,               # deterministic
                max_new_tokens=1,              # we only need internal steps
                use_cache=True,
            )

    tap.remove()
    print(f"[Dream] wrote similarity logs in {args.out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="logs/dream_baseline")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=4)
    ap.add_argument("--prompt", type=str, default="Explain diffusion decoding briefly.")
    ap.add_argument(
        "--local-model-dir",
        type=str,
        default="",
        help="Path to a local Dream checkpoint directory (HF-style). If omitted, will load from --model-id."
    )
    ap.add_argument(
        "--model-id",
        type=str,
        default="Dream-org/Dream-v0-Base-7B",
        help="HF repo id to use when --local-model-dir is not provided."
    )
    args = ap.parse_args()
    main(args)
