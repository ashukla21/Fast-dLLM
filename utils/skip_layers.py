# utils/skip_layers.py
from __future__ import annotations
import json
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch import nn

class SkipController:
    """
    Turns per-step layer-skip schedules into actual forward hooks that
    return identity for specified transformer blocks (i.e., "skip").
    Works with DreamDecoderLayer and LLaDA blocks that return (hidden_states, ...).
    """

    def __init__(self, layers: List[nn.Module], schedule: Dict[str, Any]):
        """
        Args:
            layers: flat list of decoder layers in depth order (0..L-1)
            schedule: JSON dict with keys:
                - "num_layers": int
                - "steps": int
                - "skip": { "<step_int>": [layer_idx, ...], ... }
        """
        self.layers = layers
        self.schedule = schedule
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

        # quick sanity
        n = len(layers)
        n_decl = int(schedule.get("num_layers", n))
        if n != n_decl:
            print(f"[SkipController] Warning: repo has {n} layers but schedule says {n_decl}. Using repo count {n}.")

    def clear(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []

    def _make_identity_hook(self, layer_idx: int):
        """
        Replace layer output with identity: output[0] := input hidden_states.
        Works for Dream/LLaDA where layer forward returns (hidden_states, ...).
        """
        def hook(_module, inp, out):
            # inp: tuple(inputs), out: whatever the layer returns
            # first arg to the layer is hidden_states
            hidden_states_in = inp[0]
            if isinstance(out, tuple):
                new0 = hidden_states_in
                return (new0,) + tuple(out[1:])
            else:
                # extremely rare, but handle tensor-only case
                return hidden_states_in
        return hook

    def apply_for_step(self, step: int):
        """
        Remove any previous hooks and apply hooks for the given step.
        """
        self.clear()
        to_skip = self.schedule.get("skip", {}).get(str(step), [])
        if not to_skip:
            return
        for li in to_skip:
            if 0 <= li < len(self.layers):
                h = self.layers[li].register_forward_hook(self._make_identity_hook(li))
                self._hooks.append(h)

    @staticmethod
    def from_json_file(layers: List[nn.Module], path: str) -> "SkipController":
        with open(path, "r") as f:
            schedule = json.load(f)
        return SkipController(layers, schedule)
