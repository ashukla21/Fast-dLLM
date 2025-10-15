# scheduling/layer_skip_controller.py
import json
from typing import Dict, List, Optional
import torch

class LayerSkipController:
    """
    JSON schedule like:
      { "0": [0,1,2], "1": [], "42": [3,5,7] }
    """
    def __init__(self, schedule_path: str):
        with open(schedule_path, "r") as f:
            self.table: Dict[str, List[int]] = json.load(f)

    def layers_for_step(self, step: int, num_layers: Optional[int] = None) -> List[int]:
        li = self.table.get(str(step), [])
        if num_layers is None:
            return li
        return [i for i in li if 0 <= i < num_layers]

    def get_mask(self, step_idx: int, total_steps: int, num_layers: int) -> torch.Tensor:
        """
        Return a boolean mask of size [num_layers], True = skip this layer at this step.
        Uses the JSON schedule by default.
        """
        skips = self.layers_for_step(step_idx, num_layers)
        mask = torch.zeros(num_layers, dtype=torch.bool)
        if skips:
            mask[torch.tensor(skips, dtype=torch.long)] = True
        return mask
