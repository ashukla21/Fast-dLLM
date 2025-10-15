#!/usr/bin/env python3
import argparse, os, sys, json, time
from pathlib import Path
import torch

import torch
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# --- make repo root importable ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# project imports (all local)
from scheduling.layer_skip_controller import LayerSkipController
from llada.model.modeling_llada import LLaDAModelLM, make_pre_step
from llada.model.configuration_llada import LLaDAConfig

# Optional: if LLaDA sampler reuses Dream's generation config
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
    # tokenizer source (HF id or local folder you already have; reuse Dream tokenizer path)
    p.add_argument("--tokenizer-id", type=str, required=True,
                   help="HF repo id or local folder for tokenizer (e.g., your Dream checkpoint folder)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    # 1) Tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_id, trust_remote_code=True)

    # 2) Build LLaDA config locally (no checkpoint). Ensure RoPE is enabled.
    cfg = LLaDAConfig()
    cfg.d_model = 2048
    cfg.n_heads = 16
    cfg.effective_n_kv_heads = 16
    cfg.n_layers = 16
    cfg.rope = True
    cfg.vocab_size = len(tok)
    cfg.embedding_size = max(cfg.vocab_size, 4096)  # prevent vocab>embed crash
    cfg.d_model = 4096
    cfg.n_heads = 32
    cfg.effective_n_kv_heads = 32
    cfg.n_layers = 32
    cfg.block_type = "llama"
    cfg.use_cache = False
    cfg.max_sequence_length = getattr(tok, "model_max_length", 4096) or 4096

    # 3) Instantiate model from config (no from_pretrained)
    model = LLaDAModelLM(cfg).to(device).eval()

    # 4) Inputs
    if hasattr(tok, "apply_chat_template"):
        messages = [{"role": "user", "content": args.prompt}]
        prompt_txt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_ids = tok(prompt_txt, return_tensors="pt").input_ids.to(device)
    else:
        input_ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)

    # 5) Gen config/kwargs (use Dream GenCfg if available, else kwargs)
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        mask_token_id=args.mask_token_id,
        return_dict_in_generate=True,
        output_history=args.save_history,
        threshold=args.threshold,
        torch_dtype=dtype,
    )
    gen_cfg = GenCfg(**gen_kwargs) if GenCfg is not None else None

    # 6) Layer-skip controller + pre-step callback
    schedule_path = Path(args.schedule_path).expanduser().resolve()
    ctrl = LayerSkipController(str(schedule_path))

    try:
        num_layers = model.model.config.n_layers
    except Exception:
        num_layers = getattr(model.config, "n_layers",
                             getattr(model.config, "num_hidden_layers", None))
        if num_layers is None:
            raise RuntimeError("Cannot infer number of layers for LLaDA.")
    pre_step = make_pre_step(model, num_layers)(ctrl)

    # 7) Run diffusion-style generation
    if hasattr(model, "diffusion_generate"):
        out = model.diffusion_generate(
            inputs=input_ids,
            generation_config=gen_cfg,
            pre_step_callback=pre_step,
            **({} if gen_cfg is not None else gen_kwargs),
        )
        sequences = out.sequences if hasattr(out, "sequences") else out
    else:
        raise AttributeError(
            "LLaDAModelLM has no `diffusion_generate`. "
            "Add it in modeling_llada.py and have it accept `pre_step_callback`, "
            "then call self.forward(..., layer_skip_mask=self._current_layer_skip_mask)."
        )

    # 8) Decode + print
    text = tok.decode(sequences[0], skip_special_tokens=True)
    print(text)

    # 9) Optional save
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
