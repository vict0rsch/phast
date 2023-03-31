import os
import sys
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Dict, List

import torch

try:
    import ipdb  # noqa: F401

    os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
except:  # noqa: E722
    print(
        "`ipdb` is not installed. ",
        "Consider `pip install ipdb` to improve your debugging experience.",
    )

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
