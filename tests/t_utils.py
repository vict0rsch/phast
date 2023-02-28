import sys
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Dict, List

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
DATA = None


def make_parameterization(init_search_space: dict) -> List[Dict]:
    return [
        {k: v for k, v in zip(init_search_space.keys(), prod_tuple)}
        for prod_tuple in product(*init_search_space.values())
    ]


def get_data():
    global DATA
    if DATA is None:
        data_path = (
            Path(__file__).resolve().parent.parent
            / "examples"
            / "data"
            / "is2re_bs3.pt"
        )
        DATA = torch.load(data_path)

    return deepcopy(DATA)
