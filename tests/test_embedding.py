import pytest
import torch

import t_utils as tu
from phast import embedding as phe

init_search_space = {
    "z_emb_size": [0, 8],
    "tag_emb_size": [0, 8],
    "period_emb_size": [0, 8],
    "group_emb_size": [0, 8],
    "properties": [[], phe.PhysRef.default_properties],
    "properties_proj_size": [0, 8],
    "properties_grad": [True, False],
    "final_proj_size": [0, 8],
    "n_elements": [85],
}

all_kwargs = tu.make_parameterization(init_search_space)


@pytest.mark.parametrize("kwargs", all_kwargs)
def test_inits(kwargs):
    print("kwargs: ", kwargs)
    if (kwargs["properties_proj_size"] > 0) and not kwargs["properties"]:
        with pytest.raises(ValueError):
            _ = phe.PhysEmbedding(**kwargs)
    elif (
        sum([v for k, v in kwargs.items() if "emb_size" in k]) == 0
        and not kwargs["properties"]
    ):
        with pytest.raises(ValueError):
            _ = phe.PhysEmbedding(**kwargs)
    else:
        _ = phe.PhysEmbedding(**kwargs)


@pytest.mark.parametrize("kwargs", all_kwargs)
def test_forwards(kwargs):
    sample = torch.randint(0, 85, (2, 7))
    tag = None
    print("kwargs: ", kwargs)
    if (kwargs["properties_proj_size"] > 0) and not kwargs["properties"]:
        return
    elif (
        sum([v for k, v in kwargs.items() if "emb_size" in k]) == 0
        and not kwargs["properties"]
    ):
        return

    embed = phe.PhysEmbedding(**kwargs)
    if kwargs["tag_emb_size"] > 0:
        with pytest.raises(AssertionError):
            _ = embed(sample, tag)
        tag = torch.randint(0, 2, (2, 7))
    _ = embed(sample, tag)


@pytest.mark.parametrize("kwargs", all_kwargs)
def test_shapes(kwargs):
    sample = torch.randint(0, 85, (2, 7))
    tag = None
    print("kwargs: ", kwargs)
    if (kwargs["properties_proj_size"] > 0) and not kwargs["properties"]:
        return
    elif (
        sum([v for k, v in kwargs.items() if "emb_size" in k]) == 0
        and not kwargs["properties"]
    ):
        return

    embed = phe.PhysEmbedding(**kwargs)
    if kwargs["tag_emb_size"] > 0:
        tag = torch.randint(0, 2, (2, 7))
    y = embed(sample, tag)

    if kwargs["final_proj_size"] > 0:
        assert y.shape == (*sample.shape, kwargs["final_proj_size"])
    else:
        assert y.shape == (
            *sample.shape,
            sum(
                [
                    v
                    for k, v in kwargs.items()
                    if "emb_size" in k or k == "properties_proj_size"
                ]
            )
            + (
                len(kwargs["properties"])
                * (1 - int(bool(kwargs["properties_proj_size"])))
            ),
        )


def test_repr():
    embed = phe.PhysEmbedding()
    print(embed)
    print(repr(embed))
