# scripts/run_llada.py
import argparse
from pathlib import Path
import torch

from transformers import AutoTokenizer

# LLaDA model wrapper (HF PreTrainedModel)
from fastdllm.modeling_llada import LLaDAModelLM, make_pre_step
# Layer-skip schedule controller
from fastdllm.scheduling.layer_skip_controller import LayerSkipController

# If your LLaDA diffusion API reuses the Dream generation config/mixin, this import will work.
# If you created a separate config for LLaDA, swap the import below accordingly.
try:
    from fastdllm.generation_utils import DreamGenerationConfig as _GenCfg  # shared config
except Exception:
    _GenCfg = None


def build_inputs(tokenizer, prompt: str, device: str):
    """Basic prompt → input_ids (adjust to your data format as needed)."""
    # If you use chat templates, uncomment:
    # messages = [{"role": "user", "content": prompt}]
    # prompt_txt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    # return tokenizer(prompt_txt, return_tensors="pt").input_ids.to(device)

    return tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(device)


def main():
    p = argparse.ArgumentParser("LLaDA runner with layer-skipping")
    p.add_argument("--model", type=str, required=True,
                   help="HF model id or local path for LLaDA (the HF wrapper LLaDAModelLM will be used)")
    p.add_argument("--schedule-path", type=str, required=True,
                   help="Path to skip schedule (json/csv) produced in your Step 2")
    p.add_argument("--prompt", type=str, default="Explain attention in one paragraph.")
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--mask-token-id", type=int, required=True)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--threshold", type=float, default=0.9,
                   help="If your LLaDA sampler supports confidence_threshold mode")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    # 1) Load model + tokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = LLaDAModelLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device).eval()

    # 2) Build inputs
    input_ids = build_inputs(tok, args.prompt, device)

    # 3) Build generation config/kwargs
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        mask_token_id=args.mask_token_id,
        return_dict_in_generate=True,
        output_history=False,
        threshold=args.threshold,
    )

    # Prefer a config object if your LLaDA diffusion_generate expects it,
    # otherwise pass kwargs directly.
    gen_cfg = _GenCfg(**gen_kwargs) if _GenCfg is not None else None

    # 4) Layer-skip controller + pre-step callback
    schedule_path = Path(args.schedule_path).expanduser().resolve()
    ctrl = LayerSkipController(str(schedule_path))

    # make_pre_step should be defined in fastdllm.modeling_llada analogous to Dream:
    #   def make_pre_step(model, num_layers): return lambda ctrl: callback
    try:
        num_layers = model.model.config.n_layers  # LLaDAModel -> ModelConfig
    except Exception:
        # fallback: some configs use num_hidden_layers
        num_layers = getattr(model.config, "n_layers", getattr(model.config, "num_hidden_layers", None))
        if num_layers is None:
            raise RuntimeError("Cannot infer number of layers from model config.")
    llada_pre_step = make_pre_step(model, num_layers)(ctrl)

    # 5) Run diffusion-style generation on LLaDA
    # Your Step-3 implementation should expose a method that accepts pre_step_callback.
    # Commonly we've mirrored Dream's API: model.diffusion_generate(...)
    try:
        out = model.diffusion_generate(
            inputs=input_ids,
            generation_config=gen_cfg,
            pre_step_callback=llada_pre_step,
            **({} if gen_cfg is not None else gen_kwargs),  # pass raw kwargs if no config object
        )
        sequences = out.sequences
    except AttributeError:
        # If your LLaDA diffusion lives in a separate helper, import and call it here instead.
        # Example (adjust import to your project):
        # from fastdllm.llada_generation import diffusion_generate_llada
        # out = diffusion_generate_llada(model, input_ids, pre_step_callback=llada_pre_step, **gen_kwargs)
        # sequences = out["sequences"]
        raise AttributeError(
            "model.diffusion_generate not found on LLaDA model. "
            "Ensure your LLaDA diffusion sampler exposes a callable with `pre_step_callback` "
            "or update this script to import the correct entrypoint."
        )

    # 6) Decode & print
    print(tok.decode(sequences[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
