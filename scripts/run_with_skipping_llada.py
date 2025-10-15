# scripts/run_with_skipping_llada.py
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
from llada.model.modeling_llada import LLaDAModelLM
from llada.generate import generate_with_dual_cache  # we’ll add pre/post callbacks

def get_decoder_layers_llada(hf_model):
    if hasattr(hf_model, "model"):
        core = hf_model.model
    else:
        core = hf_model
    tr = getattr(core, "transformer", None)
    if tr is None:
        raise RuntimeError("LLaDA core model has no `transformer` module")
    if hasattr(tr, "blocks"):
        layers = list(tr.blocks)
        if layers: return layers
    if hasattr(tr, "block_groups"):
        layers = []
        for bg in tr.block_groups:
            layers.extend(list(bg))
        if layers: return layers
    layers = [m for m in core.modules() if m.__class__.__name__.startswith("LLaDA") and "Block" in m.__class__.__name__]
    if layers:
        return layers
    raise RuntimeError("Could not find LLaDA decoder layers")

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float16

    model_id = "GSAI-ML/LLaDA-8B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = LLaDAModelLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=dtype).to(device).eval()

    layers = get_decoder_layers_llada(model)
    print(f"[LLaDA] depth = {len(layers)}")

    sc = SkipController.from_json_file(layers, args.skip_schedule)
    steps_to_skip = sorted(map(int, sc.schedule.get("skip", {}).keys()))
    print(f"[LLaDA] skipping on steps: {steps_to_skip}")

    def pre_step(step: int):
        sc.apply_for_step(step)

    def post_step(step: int):
        sc.clear()

    # Build prompt
    inputs = tok(args.prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        _ = generate_with_dual_cache(
            model=model,
            prompt=inputs["input_ids"],
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_size,
            temperature=0.0,
            remasking='low_confidence',
            step_callback=None,           # we’ll expose pre/post in llada.generate (see patch below)
            pre_step_callback=pre_step,   # NEW
            post_step_callback=post_step, # NEW
        )

    print(f"[LLaDA] done. Skipping ran with schedule: {os.path.basename(args.skip_schedule)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="logs/llada_skip")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--gen-length", type=int, default=64)
    ap.add_argument("--prompt", type=str, default="Say hello.")
    ap.add_argument("--skip-schedule", type=str, required=True)
    args = ap.parse_args()
    main(args)
