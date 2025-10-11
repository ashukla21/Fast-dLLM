from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os, csv, torch, argparse
from transformers import AutoTokenizer
from utils.similarity import LayerTap, cosine_matrix, cosine_diag

# import Dream model (adjust import path if your repo differs)
from dream.model.modeling_dream import DreamModel

def get_decoder_layers_dream(model):
    """
    Return a flat list of Dream's decoder blocks. We try a few layouts:
    - model.transformer.blocks (common)
    - model.transformer.block_groups[*] (grouped layout)
    - fallback: any module whose class name ends with 'Block' or 'DecoderLayer'
    """
    layers = []

    # Try common container locations
    core = getattr(model, "model", model)  # unwrap if wrapped
    tr = getattr(core, "transformer", None)

    if tr is not None and hasattr(tr, "blocks"):
        layers = list(tr.blocks)
        if layers:
            return layers

    if tr is not None and hasattr(tr, "block_groups"):
        tmp = []
        for bg in tr.block_groups:  # nn.ModuleList
            tmp.extend(list(bg))
        if tmp:
            return tmp

    # Fallback: scan modules
    for name, mod in core.named_modules():
        cls = type(mod).__name__
        if cls.endswith("DecoderLayer") or cls.endswith("Block"):
            layers.append(mod)
    if not layers:
        raise RuntimeError("Could not find Dream decoder layers (checked blocks, block_groups, and fallbacks).")
    return layers

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # bf16 on GPU if available, else fp16 on CPU to save memory
    if device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    # ---- model + tokenizer ----
    # IMPORTANT: model_path must point to an actual checkpoint directory (HF-style)
    # with config + weights (not just the source code folder).
    model_path = "./dream"  # change this to a real checkpoint folder if needed
    if not (Path(model_path) / "config.json").exists():
        raise FileNotFoundError(
            f"`{model_path}` does not look like a checkpoint directory (missing config.json). "
            "Point `model_path` to a valid Dream checkpoint (local HF-style folder)."
        )

    print(f"Loading Dream model from {model_path}...")
    model = DreamModel.from_pretrained(model_path, torch_dtype=dtype).to(device)
    model.eval()

    tok = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)

    layers = get_decoder_layers_dream(model)
    tap = LayerTap(layers, pool="last")

    prev = None

    across_path = os.path.join(args.out_dir, "across_steps_cosine.csv")

    with open(across_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step"] + [f"layer_{i}" for i in range(len(layers))])

        def on_step(step):
            nonlocal prev
            A = tap.stacked()  # [L, H] pooled activations for this step
            torch.save(
                cosine_matrix(A),
                os.path.join(args.out_dir, f"within_step_cosine_step{step}.pt"),
            )
            if prev is not None:
                w.writerow([step] + cosine_diag(prev, A).tolist())
            prev = A
            tap.clear()

        inputs = tok(args.prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model.diffusion_generate(
                inputs["input_ids"],
                step_callback=on_step,         # our hook
                steps=args.steps,              # num diffusion steps
                block_length=args.block_size,  # block size we log against
                temperature=0.0,               # deterministic
                max_new_tokens=1,              # minimal decode to trigger steps
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
    ap.add_argument("--gen-length", type=int, default=64)  # kept for CLI symmetry; unused here
    args = ap.parse_args()
    main(args)
