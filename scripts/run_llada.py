#!/usr/bin/env python3
import argparse, os, sys, json, time
from pathlib import Path
import torch

# --- make repo root importable ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# project imports (all local to your repo)
from scheduling.layer_skip_controller import LayerSkipController
from llada.model.modeling_llada import LLaDAModelLM, make_pre_step
from llada.model.configuration_llada import LLaDAConfig

# Optional: if your LLaDA sampler reuses Dream's generation config
try:
    from dream.model.generation_utils import DreamGenerationConfig as GenCfg
except Exception:
    GenCfg = None  # fall back to kwargs

def main():
    p = argparse.ArgumentParser("LLaDA diffusion runner with layer skipping (local repo model)")
    # IO / logging
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--save-history", action="store_true")
    # generation
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--mask-token-id", type=int, required=True)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    # prompt + schedule
    p.add_argument("--prompt", type=str, default="Explain attention in one paragraph.")
    p.add_argument("--schedule-path", type=str, required=True)
    # tokenizer source (HF id or local folder you already have; e.g. reuse Dream tokenizer path)
    p.add_argument("--tokenizer-id", type=str, required=True,
                   help="HF repo id or local folder for tokenizer (can reuse your Dream checkpoint folder)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    # 1) Tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_id, trust_remote_code=True)

    # 2) Build model from local code (no from_pretrained)
    #    If your config needs non-default values, set them here.
    cfg = LLaDAConfig()
    model = LLaDAModelLM(cfg).to(device).eval()

    # 3) Inputs
    if hasattr(tok, "apply_chat_template"):
        messages = [{"role": "user", "content": args.prompt}]
        prompt_txt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_ids = tok(prompt_txt, return_tensors="pt").input_ids.to(device)
    else:
        input_ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)

    # 4) Gen config/kwargs (use Dream GenCfg if available, otherwise kwargs)
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        mask_token_id=args.mask_token_id,
        return_dict_in_generate=True,
        output_history=args.save_history,
        threshold=args.threshold,  # only used if your sampler supports it
        torch_dtype=dtype,
    )
    gen_cfg = GenCfg(**gen_kwargs) if GenCfg is not None else None

    # 5) Layer-skip controller + pre-step callback
    schedule_path = Path(args.schedule_path).expanduser().resolve()
    ctrl = LayerSkipController(str(schedule_path))

    # infer number of layers (LLaDAConfig uses n_layers)
    try:
        num_layers = model.model.config.n_layers
    except Exception:
        num_layers = getattr(model.config, "n_layers",
                      getattr(model.config, "num_hidden_layers", None))
        if num_layers is None:
            raise RuntimeError("Cannot infer number of layers for LLaDA.")
    llada_pre_step = make_pre_step(model, num_layers)(ctrl)

    # 6) Run diffusion-style generation
    if hasattr(model, "diffusion_generate"):
        out = model.diffusion_generate(
            inputs=input_ids,
            generation_config=gen_cfg,
            pre_step_callback=llada_pre_step,
            **({} if gen_cfg is not None else gen_kwargs),
        )
        sequences = out.sequences if hasattr(out, "sequences") else out
    else:
        raise AttributeError(
            "LLaDAModelLM has no `diffusion_generate`. "
            "Expose a sampler entrypoint (e.g., `diffusion_generate`) that accepts `pre_step_callback`, "
            "or import the correct generation function here."
        )

    # 7) Decode + print
    text = tok.decode(sequences[0], skip_special_tokens=True)
    print(text)

    # 8) Optional save
    if args.output_dir:
        outdir = Path(args.output_dir).expanduser()
        outdir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        (outdir / f"output_{stamp}.txt").write_text(text)
        if args.save_history and getattr(out, "history", None) is not None:
            try:
                torch.save(torch.stack(out.history), outdir / f"history_{stamp}.pt")
            except Exception:
                pass
        meta = {
            "prompt": args.prompt,
            "schedule_path": str(schedule_path),
            "steps": args.steps,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "mask_token_id": args.mask_token_id,
            "dtype": args.dtype,
            "threshold": args.threshold,
            "tokenizer_id": args.tokenizer_id,
        }
        (outdir / f"meta_{stamp}.json").write_text(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
