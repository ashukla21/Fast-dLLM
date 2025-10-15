# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

from typing import Optional, Callable

import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer
# IMPORTANT: use package-relative import so running from repo root works.
from .model.modeling_llada import LLaDAModelLM


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    The Gumbel max is a method for sampling categorical distributions.
    For MDM, low-precision Gumbel Max hurts quality, so we use float64.
    """
    if temperature == 0:
        return logits
    logits64 = logits.to(torch.float64)
    noise = torch.rand_like(logits64, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits64.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    In the reverse process, [0, 1] is uniformly discretized into `steps`.
    With a linear noise schedule, the expected # of transitioned tokens
    per step is consistent. Precompute those counts for each batch item.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)  # [B, 1]
    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    ) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens

from fastdllm.scheduling.layer_skip_controller import LayerSkipController

def make_pre_step_llada(model, ctrl):
    model._layer_skip_mask = None
    num_layers = model.config.n_layers
    def _cb(step_idx: int):
        to_skip = ctrl.layers_for_step(step_idx, num_layers=num_layers)
        mask = torch.zeros(num_layers, dtype=torch.bool, device=model.device)
        if to_skip:
            mask[to_skip] = True
        model._layer_skip_mask = mask
    return _cb

@torch.no_grad()
def generate(
    model,
    prompt: torch.Tensor,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    threshold: Optional[float] = None,
    factor: Optional[float] = None,
    step_callback: Optional[Callable[[int], None]] = None,
    pre_step_callback: Optional[Callable[[int], None]] = None,
    post_step_callback: Optional[Callable[[int, torch.Tensor, torch.Tensor], None]] = None,
    generation_logits_hook: Optional[Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    generation_tokens_hook: Optional[Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
):
    """
    Vanilla LLaDA diffusion decoding (no cache).

    Args:
        model: Mask predictor (LLaDAModelLM).
        prompt: Tensor (B, L).
        steps: Total sampling steps (<= gen_length).
        gen_length: Generated answer length.
        block_length: If < gen_length, semi-autoregressive remasking in blocks.
        temperature: Sampling temperature for categorical noise.
        remasking: 'low_confidence' or 'random'.
        mask_id: Token id of [MASK] (126336 for LLaDA-8B).
    """
    if generation_logits_hook is None:
        generation_logits_hook = lambda i, x, logits: logits
    if generation_tokens_hook is None:
        generation_tokens_hook = lambda i, x, logits: x
    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=model.device,
    )
    x[:, : prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    nfe = 0
    step_idx = 0  # global diffusion-step counter across all blocks

    for num_block in range(num_blocks):
        start = prompt.shape[1] + num_block * block_length
        end = start + block_length

        block_mask_index = (x[:, start:end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        i = 0

        while True:
            nfe += 1

            # (1) pre-step hook (e.g., apply layer-skip schedule)
            if callable(pre_step_callback):
                try:
                    pre_step_callback(step_idx)
                except Exception as _e:
                    print(f"[LLaDA] pre_step_callback error at step {step_idx}: {_e}")

            # (2) forward -> logits
            mask_index = (x == mask_id)
            logits = model(x, layer_skip_mask=getattr(model, "_layer_skip_mask", None),).logits
            logits = generation_logits_hook(step_idx, x, logits)

            # respect block boundary
            mask_index[:, end:] = 0

            # (3) update x using your existing transfer logic
            if factor is None:
                x0, transfer_index = get_transfer_index(
                    logits,
                    temperature,
                    remasking,
                    mask_index,
                    x,
                    num_transfer_tokens[:, i] if threshold is None else None,
                    threshold,
                )
            else:
                x0, transfer_index = get_transfer_index_dynamic(
                    logits, temperature, remasking, mask_index, x, None, factor
                )
            x[transfer_index] = x0[transfer_index]

            # (4) post-update token hook (lets you rewrite x if needed)
            x = generation_tokens_hook(step_idx, x, logits)

            # (5) post-step hook (e.g., log, collect stats)
            if callable(post_step_callback):
                try:
                    post_step_callback(step_idx, x, logits)
                except Exception as _e:
                    print(f"[LLaDA] post_step_callback error at step {step_idx}: {_e}")

            # (6) existing step callback
            if callable(step_callback):
                step_callback(step_idx)
            step_idx += 1

            i += 1
            if (x[:, start:end] == mask_id).sum() == 0:
                break

    return x, nfe


@torch.no_grad()
def generate_with_prefix_cache(
    model,
    prompt: torch.Tensor,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    threshold: Optional[float] = None,
    factor: Optional[float] = None,
    step_callback: Optional[Callable[[int], None]] = None,
    pre_step_callback: Optional[Callable[[int], None]] = None,
    post_step_callback: Optional[Callable[[int, torch.Tensor, torch.Tensor], None]] = None,
    generation_logits_hook: Optional[Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    generation_tokens_hook: Optional[Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
):
    """
    LLaDA decoding with a prefix KV-cache.
    """
    if generation_logits_hook is None:
        generation_logits_hook = lambda i, x, logits: logits
    if generation_tokens_hook is None:
        generation_tokens_hook = lambda i, x, logits: x
    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=model.device,
    )
    x[:, : prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    nfe = 0
    step_idx = 0

    for num_block in range(num_blocks):
        start = prompt.shape[1] + num_block * block_length
        end = start + block_length

        block_mask_index = (x[:, start:end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        output = model(x, use_cache=True)
        # (1) pre-step
        if callable(pre_step_callback):
            try:
                pre_step_callback(step_idx)
            except Exception as _e:
                print(f"[LLaDA] pre_step_callback error at step {step_idx}: {_e}")

        mask_index = (x == mask_id)
        mask_index[:, end:] = 0
        logits = output.logits
        logits = generation_logits_hook(step_idx, x, logits)

        if factor is None:
            x0, transfer_index = get_transfer_index(
                logits,
                temperature,
                remasking,
                mask_index,
                x,
                num_transfer_tokens[:, 0] if threshold is None else None,
                threshold,
            )
        else:
            x0, transfer_index = get_transfer_index_dynamic(
                logits, temperature, remasking, mask_index, x, None, factor
            )
        x[transfer_index] = x0[transfer_index]
        x = generation_tokens_hook(step_idx, x, logits)

        if callable(post_step_callback):
            try:
                post_step_callback(step_idx, x, logits)
            except Exception as _e:
                print(f"[LLaDA] post_step_callback error at step {step_idx}: {_e}")

        if callable(step_callback):
            step_callback(step_idx)
        step_idx += 1

        past_key_values = output.past_key_values

        # keep only prefix cache
        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :start],)
        past_key_values = new_past_key_values

        nfe += 1
        i = 1

        while True:
            if (x[:, start:end] == mask_id).sum() == 0:
                break

            nfe += 1

            if callable(pre_step_callback):
                try:
                    pre_step_callback(step_idx)
                except Exception as _e:
                    print(f"[LLaDA] pre_step_callback error at step {step_idx}: {_e}")

            mask_index = (x[:, start:] == mask_id)
            mask_index[:, block_length:] = 0

            out = model(x[:, start:], past_key_values=past_key_values, use_cache=True)
            logits = out.logits
            logits = generation_logits_hook(step_idx, x[:, start:], logits)

            # (optional) leave your gumbel line as-is if you like:
            # logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            # _ = torch.argmax(logits_with_noise, dim=-1)

            if factor is None:
                x0, transfer_index = get_transfer_index(
                    logits,
                    temperature,
                    remasking,
                    mask_index,
                    x[:, start:],
                    num_transfer_tokens[:, i] if threshold is None else None,
                    threshold,
                )
            else:
                x0, transfer_index = get_transfer_index_dynamic(
                    logits, temperature, remasking, mask_index, x[:, start:], None, factor
                )

            x[:, start:][transfer_index] = x0[transfer_index]
            x = generation_tokens_hook(step_idx, x, logits)

            if callable(post_step_callback):
                try:
                    post_step_callback(step_idx, x, logits)
                except Exception as _e:
                    print(f"[LLaDA] post_step_callback error at step {step_idx}: {_e}")

            if callable(step_callback):
                step_callback(step_idx)
            step_idx += 1

            i += 1

    return x, nfe


@torch.no_grad()
def generate_with_dual_cache(
    model,
    prompt: torch.Tensor,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    threshold: Optional[float] = None,
    factor: Optional[float] = None,
    step_callback: Optional[Callable[[int], None]] = None,
    pre_step_callback: Optional[Callable[[int], None]] = None,
    post_step_callback: Optional[Callable[[int, torch.Tensor, torch.Tensor], None]] = None,
    generation_logits_hook: Optional[Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    generation_tokens_hook: Optional[Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
):
    """
    LLaDA decoding with a dual-cache strategy (cache window for current block).
    """
    if generation_logits_hook is None:
        generation_logits_hook = lambda i, x, logits: logits
    if generation_tokens_hook is None:
        generation_tokens_hook = lambda i, x, logits: x
    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=model.device,
    )
    x[:, : prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    nfe = 0
    step_idx = 0

    for num_block in range(num_blocks):
        start = prompt.shape[1] + num_block * block_length
        end = start + block_length

        block_mask_index = (x[:, start:end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        # cache init + first update
        output = model(x, use_cache=True)
        # (1) pre-step
        if callable(pre_step_callback):
            try:
                pre_step_callback(step_idx)
            except Exception as _e:
                print(f"[LLaDA] pre_step_callback error at step {step_idx}: {_e}")

        mask_index = (x == mask_id)
        mask_index[:, end:] = 0
        logits = output.logits
        logits = generation_logits_hook(step_idx, x, logits)

        if factor is None:
            x0, transfer_index = get_transfer_index(
                logits,
                temperature,
                remasking,
                mask_index,
                x,
                num_transfer_tokens[:, 0] if threshold is None else None,
                threshold,
            )
        else:
            x0, transfer_index = get_transfer_index_dynamic(
                logits, temperature, remasking, mask_index, x, None, factor
            )
        x[transfer_index] = x0[transfer_index]
        x = generation_tokens_hook(step_idx, x, logits)

        if callable(post_step_callback):
            try:
                post_step_callback(step_idx, x, logits)
            except Exception as _e:
                print(f"[LLaDA] post_step_callback error at step {step_idx}: {_e}")

        if callable(step_callback):
            step_callback(step_idx)
        step_idx += 1

        past_key_values = output.past_key_values

        # keep only prefix cache
        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :start],)
        past_key_values = new_past_key_values

        nfe += 1
        i = 1
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, start:end] = 1

        while True:
            if (x[:, start:end] == mask_id).sum() == 0:
                break

        nfe += 1

        if callable(pre_step_callback):
            try:
                pre_step_callback(step_idx)
            except Exception as _e:
                print(f"[LLaDA] pre_step_callback error at step {step_idx}: {_e}")

        # Forward only on the current block (window) with dual cache
        out = model(
            x[:, start:end],
            past_key_values=past_key_values,
            use_cache=True,
            replace_position=replace_position,
        )
        logits = out.logits                              # [B, block_length, V]
        logits = generation_logits_hook(step_idx, x[:, start:end], logits)

        # Window-local mask
        mask_index = (x[:, start:end] == mask_id)        # [B, block_length]

        # --- Single, window-local transfer ---
        if factor is None:
            x0, transfer_index = get_transfer_index(
                logits,
                temperature,
                remasking,
                mask_index,
                x[:, start:end],                         # pass window x
                num_transfer_tokens[:, i] if threshold is None else None,
                threshold,
            )
        else:
            x0, transfer_index = get_transfer_index_dynamic(
                logits, temperature, remasking, mask_index, x[:, start:end], None, factor
            )

        # Write back only into the window
        x[:, start:end][transfer_index] = x0[transfer_index]

        # Token-level post-edit hook (you can pass full x; logits are window-sized)
        x = generation_tokens_hook(step_idx, x, logits)

        if callable(post_step_callback):
            try:
                post_step_callback(step_idx, x, logits)
            except Exception as _e:
                print(f"[LLaDA] post_step_callback error at step {step_idx}: {_e}")

        if callable(step_callback):
            step_callback(step_idx)
        step_idx += 1
        i += 1

    return x, nfe


def get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    num_transfer_tokens: Optional[torch.Tensor],
    threshold: Optional[float] = None,
):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, L]

    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # [B, L]
    elif remasking == "random":
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index


def get_transfer_index_dynamic(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    num_transfer_tokens,  # unused in dynamic path
    factor: float = 1,
):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, L]

    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # [B, L]
    elif remasking == "random":
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)

    for j in range(confidence.shape[0]):
        ns = list(range(1, num_transfer_tokens[j] + 1))
        es = [factor / (n + 1) for n in ns]
        threshs = [1 - e for e in es]

        # always transfer at least one token
        threshs[0] = -1
        sorted_conf = torch.sort(confidence[j][mask_index[j]], dim=-1, descending=True)[0]
        assert len(sorted_conf) == len(threshs)
        for top_i in range(len(threshs)):
            if sorted_conf[top_i] < threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs) - 1:
            top_i += 1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index


def main():
    # simple demo; adjust device/dtype as needed
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = (
        LLaDAModelLM.from_pretrained(
            "GSAI-ML/LLaDA-8B-Instruct",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"
    m = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)["input_ids"]
    input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)

    def on_step(i: int):
        if i % 10 == 0:
            print(f"[step] {i}")

    out, _ = generate_with_dual_cache(
        model,
        input_ids,
        steps=128,
        gen_length=128,
        block_length=32,
        temperature=0.0,
        remasking="low_confidence",
        step_callback=on_step,
    )
    print(tokenizer.batch_decode(out[:, input_ids.shape[1] :], skip_special_tokens=True)[0])


if __name__ == "__main__":
    main()
