# scripts/run_dream.py
import argparse
from pathlib import Path
import torch

from fastdllm.scheduling.layer_skip_controller import LayerSkipController
from fastdllm.modeling_dream import DreamModel, DreamGenerationConfig, make_pre_step
# ^ adjust imports to your actual package layout

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Dream-7B", help="HF model id or local path")
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--mask-token-id", type=int, required=True)
    p.add_argument("--schedule-path", type=str, required=True,
                   help="Path to skip schedule file (json/csv) you produced in Step 2")
    p.add_argument("--prompt", type=str, default="Hello world")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load model
    model = DreamModel.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    # 2) Tokenize prompt (replace with your tokenizer)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    m = [{"role": "user", "content": args.prompt}]
    prompt_txt = tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    input_ids = tok(prompt_txt, return_tensors="pt").input_ids.to(device)

    # 3) Gen config (diffusion params)
    gen_cfg = DreamGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        mask_token_id=args.mask_token_id,
        return_dict_in_generate=True,
        output_history=False,
    )

    # 4) Layer-skip controller + pre-step callback
    schedule_path = Path(args.schedule_path).expanduser().resolve()
    ctrl = LayerSkipController(str(schedule_path))
    dream_pre_step = make_pre_step(model, model.config.num_hidden_layers)(ctrl)

    # 5) Run diffusion generate with the pre_step_callback
    out = model.diffusion_generate(
        inputs=input_ids,
        generation_config=gen_cfg,
        pre_step_callback=dream_pre_step,   # ← key line
    )

    seq = out.sequences  # [B, total_len]
    print(tok.decode(seq[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
