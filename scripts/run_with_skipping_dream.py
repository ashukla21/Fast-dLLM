# scripts/run_with_skipping_dream.py
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import argparse
import torch
from transformers import AutoTokenizer

from utils.skip_layers import SkipController
from utils.similarity import LayerTap  # optional, just to easily fetch layers
from dream.model.modeling_dream import DreamModel

# --- helpers to collect Dream layers (same logic you used)
def get_decoder_layers_dream(model):
    layers = []
    for _, mod in model.named_modules():
        if type(mod).__name__ in {"DreamDecoderLayer"}:
            layers.append(mod)
    if not layers:
        for _, mod in model.named_modules():
            cls = type(mod).__name__
            if "Dream" in cls and "Block" in cls:
                layers.append(mod)
    if not layers:
        raise RuntimeError("Could not find Dream decoder layers")
    return layers

def build_model(device, dtype, local_model_dir: str|None, hf_repo_id: str):
    if local_model_dir:
        print(f"[Dream] Loading from {local_model_dir}")
        model = DreamModel.from_pretrained(local_model_dir, torch_dtype=dtype).to(device)
        tok = AutoTokenizer.from_pretrained(local_model_dir, trust_remote_code=True)
    else:
        print(f"[Dream] Loading from HF repo: {hf_repo_id}")
        model = DreamModel.from_pretrained(hf_repo_id, torch_dtype=dtype, trust_remote_code=True).to(device)
        tok = AutoTokenizer.from_pretrained(hf_repo_id, trust_remote_code=True)
    model.eval()
    return model, tok

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float16

    model, tok = build_model(device, dtype, args.local_model_dir or None, args.model_id)

    layers = get_decoder_layers_dream(model)
    print(f"[Dream] depth = {len(layers)}")

    # Build the skip controller from schedule
    sc = SkipController.from_json_file(layers, args.skip_schedule)
    steps_to_skip = sorted(map(int, sc.schedule.get("skip", {}).keys()))
    print(f"[Dream] skipping on steps: {steps_to_skip}")

    # define a PRE-step callback so skips apply to the step about to run
    def pre_step(step: int):
        sc.apply_for_step(step)

    # (optional) define a POST-step callback that clears hooks (safety)
    def post_step(step: int):
        # for this implementation, we keep hooks for the duration of the step only
        sc.clear()

    # Package pre/post into kwargs expected by Dream (we’ll add handling below)
    gen_kwargs = dict(
        steps=args.steps,
        block_length=args.block_size,
        temperature=0.0,
        max_new_tokens=1,
        use_cache=True,
        pre_step_callback=pre_step,     # requires small change in generation_utils.py
        post_step_callback=post_step,   # requires small change in generation_utils.py
    )

    inputs = tok(args.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.diffusion_generate(inputs=inputs["input_ids"], **gen_kwargs)

    print(f"[Dream] done. Skipping ran with schedule: {os.path.basename(args.skip_schedule)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="logs/dream_skip")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--prompt", type=str, default="Say hello.")
    ap.add_argument("--local-model-dir", type=str, default="")
    ap.add_argument("--model-id", type=str, default="Dream-org/Dream-v0-Base-7B")
    ap.add_argument("--skip-schedule", type=str, required=True,
                    help="Path to skip schedule JSON from make_skip_schedule.py")
    args = ap.parse_args()
    main(args)
