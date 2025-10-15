#!/usr/bin/env python3
import argparse, os, sys, json, time
from pathlib import Path
import torch
from transformers import AutoTokenizer

# --- make repo root importable ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# project imports
from scheduling.layer_skip_controller import LayerSkipController
from llada.model.modeling_llada import LLaDAModelLM, make_pre_step # ensure make_pre_step exists in modeling_llada

# Reuse Dream's generation config if your LLaDA sampler expects it
try:
    from dream.model.generation_utils import DreamGenerationConfig as GenCfg
except Exception:
    GenCfg = None  # fall back to kwargs


def _resolve_model_path(args):
    return args.local_model_dir if args.local_model_dir else args.model


def _ensure_mask_id(args, tok):
    if args.mask_token_id is not None:
        return args.mask_token_id
    mid = getattr(tok, "mask_token_id", None)
    if mid is None and hasattr(tok, "convert_tokens_to_ids"):
        for cand in ("<mask>", "<MASK>", "[MASK]", "<extra_id_0>"):
            _id = tok.convert_tokens_to_ids(cand)
            if _id is not None and _id != tok.unk_token_id:
                mid = _id
                break
    if mid is None:
        raise ValueError("--mask-token-id not provided and could not be inferred from tokenizer.")
    return int(mid)


def main():
    p = argparse.ArgumentParser("LLaDA diffusion runner with layer skipping")
    # model + io
    p.add_argument("--model", type=str, required=False, default="mlgsai/LLaDA-7B",
                   help="HF model id (ignored if --local-model-dir is set)")
    p.add_argument("--local-model-dir", type=str, default=None,
                   help="Path to local LLaDA checkpoint")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--save-history", action="store_true")
    # generation
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--mask-token-id", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    # prompt + schedule
    p.add_argument("--prompt", type=str, default="Explain attention in one paragraph.")
    p.add_argument("--schedule-path", type=str, required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model_path = _resolve_model_path(args)

    # 1) Load tokenizer & model
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    mask_token_id = _ensure_mask_id(args, tok)

    model = LLaDAModelLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True
    ).to(device).eval()

    # 2) Inputs
    if hasattr(tok, "apply_chat_template"):
        messages = [{"role": "user", "content": args.prompt}]
        prompt_txt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_ids = tok(prompt_txt, return_tensors="pt").input_ids.to(device)
    else:
        input_ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)

    # 3) Config/kwargs for diffusion sampler
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        mask_token_id=mask_token_id,
        return_dict_in_generate=True,
        output_history=args.save_history,
        threshold=args.threshold,  # only used if your sampler supports it
    )
    gen_cfg = GenCfg(**gen_kwargs) if GenCfg is not None else None

    # 4) Layer-skip controller + pre-step callback
    schedule_path = Path(args.schedule_path).expanduser().resolve()
    ctrl = LayerSkipController(str(schedule_path))

    # figure out number of layers
    try:
        num_layers = model.model.config.n_layers
    except Exception:
        num_layers = getattr(model.config, "n_layers", getattr(model.config, "num_hidden_layers", None))
        if num_layers is None:
            raise RuntimeError("Cannot infer number of layers for LLaDA.")
    llada_pre_step = make_pre_step(model, num_layers)(ctrl)

    # 5) Run diffusion-style generation
    # Your Step-3 changes should expose `diffusion_generate` on the HF wrapper.
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
            "LLaDAModelLM has no `diffusion_generate`. Ensure your LLaDA sampler is wired like Dream."
        )

    text = tok.decode(sequences[0], skip_special_tokens=True)
    print(text)

    # 6) Optional save
    if args.output_dir:
        outdir = Path(args.output_dir).expanduser()
        outdir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        (outdir / f"output_{stamp}.txt").write_text(text)
        if args.save_history and getattr(out, "history", None) is not None:
            torch.save(torch.stack(out.history), outdir / f"history_{stamp}.pt")
        meta = {
            "prompt": args.prompt,
            "model_path": str(model_path),
            "schedule_path": str(schedule_path),
            "steps": args.steps,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "mask_token_id": mask_token_id,
            "dtype": args.dtype,
            "threshold": args.threshold,
        }
        (outdir / f"meta_{stamp}.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
