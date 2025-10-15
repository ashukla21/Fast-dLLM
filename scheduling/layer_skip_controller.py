import json
from typing import Dict, List, Optional

class LayerSkipController:
    """
    Loads a JSON schedule like:
    {
      "0":  [0,1,2],   # at diffusion step 0, skip these layer indices
      "1":  [],
      "42": [3,5,7]
    }
    and answers "which layers to skip at this step?"
    """
    def __init__(self, schedule_path: str):
        with open(schedule_path, "r") as f:
            self.table: Dict[str, List[int]] = json.load(f)

    def layers_for_step(self, step: int, num_layers: Optional[int] = None) -> List[int]:
        li = self.table.get(str(step), [])
        if num_layers is None:
            return li
        # clamp to valid range just in case
        return [i for i in li if 0 <= i < num_layers]
