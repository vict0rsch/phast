try:
    import torch_geometric  # noqa: F401

    PYG_OK = True
except ImportError:
    PYG_OK = False

import torch


def ensure_pyg_ok(func):
    """
    Decorator to ensure that torch_geometric is installed when
    using a function that requires it.

    Args:
        func (callable): Function to decorate.
    """

    def wrapper(*args, **kwargs):
        if not PYG_OK:
            raise ImportError(
                "torch_geometric is not installed. "
                + "Install it to use this feature -> "
                + "https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"  # noqa: E501
            )
        return func(*args, **kwargs)

    return wrapper


def get_pbc_distances(
    pos,
    edge_index,
    cell,
    cell_offsets,
    neighbors,
    return_offsets=False,
    return_distance_vec=False,
):
    """Compute distances between atoms with periodic boundary conditions.

    From `OCP <https://github.com/Open-Catalyst-Project/ocp>`_.
    """
    row, col = edge_index

    distance_vectors = pos[row] - pos[col]

    # correct for pbc
    neighbors = neighbors.to(cell.device)
    cell = torch.repeat_interleave(cell, neighbors, dim=0)
    offsets = cell_offsets.float().view(-1, 1, 3).bmm(cell.float()).view(-1, 3)
    distance_vectors += offsets

    # compute distances
    distances = distance_vectors.norm(dim=-1)

    # redundancy: remove zero distances
    nonzero_idx = torch.arange(len(distances), device=distances.device)[distances != 0]
    edge_index = edge_index[:, nonzero_idx]
    distances = distances[nonzero_idx]

    out = {
        "edge_index": edge_index,
        "distances": distances,
    }

    if return_distance_vec:
        out["distance_vec"] = distance_vectors[nonzero_idx]

    if return_offsets:
        out["offsets"] = offsets[nonzero_idx]

    return out
