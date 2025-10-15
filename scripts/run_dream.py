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
from dream.model.modeling_dream import DreamModel, DreamGenerationConfig, make_pre_step


def _resolve_model_path(args):
    return args.local_model_dir if args.local_model_dir else args.model


def _ensure_mask_id(args, tok):
    if args.mask_token_id is not None:
        return args.mask_token_id
    mid = getattr(tok, "mask_token_id", None)
    if mid is None and hasattr(tok, "convert_tokens_to_ids"):
        # common fallbacks
        for cand in ("<mask>", "<MASK>", "[MASK]", "<extra_id_0>"):
            _id = tok.convert_tokens_to_ids(cand)
            if _id is not None and _id != tok.unk_token_id:
                mid = _id
                break
    if mid is None:
        raise ValueError(
            "--mask-token-id not provided and could not be inferred from tokenizer."
        )
    return int(mid)


def main():
    p = argparse.ArgumentParser("Dream diffusion runner with layer skipping")
    # model + io
    p.add_argument("--model", type=str, default="GSAI-ML/Dream-7B-Instruct",
                   help="HF model id (ignored if --local-model-dir is set)")
    p.add_argument("--local-model-dir", type=str, default=None,
                   help="Path to local Dream checkpoint (preferred on Colab/Drive)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Optional directory to save outputs")
    p.add_argument("--save-history", action="store_true",
                   help="Save x history (if enabled in generation config)")
    # generation
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--mask-token-id", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    # prompt + schedule
    p.add_argument("--prompt", type=str, default="Hello world")
    p.add_argument("--schedule-path", type=str, required=True,
                   help="Skip schedule produced in Step 2 (json/csv)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model_path = _resolve_model_path(args)

    # 1) Load tokenizer & model
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    mask_token_id = _ensure_mask_id(args, tok)

    model = DreamModel.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True
    ).to(device).eval()

    # 2) Build prompt → input_ids (chat template if needed)
    if hasattr(tok, "apply_chat_template"):
        messages = [{"role": "user", "content": args.prompt}]
        prompt_txt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_ids = tok(prompt_txt, return_tensors="pt").input_ids.to(device)
    else:
        input_ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)

    # 3) Generation config
    gen_cfg = DreamGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        mask_token_id=mask_token_id,
        return_dict_in_generate=True,
        output_history=args.save_history,
    )

    # 4) Layer-skip controller + callback
    schedule_path = Path(args.schedule_path).expanduser().resolve()
    ctrl = LayerSkipController(str(schedule_path))
    # NOTE: make_pre_step(model, num_layers) returns a factory; call with ctrl to get the callback
    dream_pre_step = make_pre_step(model, model.config.num_hidden_layers)(ctrl)

    # 5) Run diffusion
    out = model.diffusion_generate(
        inputs=input_ids,
        generation_config=gen_cfg,
        pre_step_callback=dream_pre_step,
    )
    sequences = out.sequences if hasattr(out, "sequences") else out
    text = tok.decode(sequences[0], skip_special_tokens=True)
    print(text)

    # 6) Optional save
    if args.output_dir:
        outdir = Path(args.output_dir).expanduser()
        outdir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        (outdir / f"output_{stamp}.txt").write_text(text)
        # history tensor (optional)
        if args.save_history and getattr(out, "history", None) is not None:
            torch.save(torch.stack(out.history), outdir / f"history_{stamp}.pt")
        # metadata
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
        }
        (outdir / f"meta_{stamp}.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
