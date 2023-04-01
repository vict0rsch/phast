"""In the context of the `OC20 dataset <https://opencatalystproject.org/index.html>`_,
rewire each 3D molecular graph according to 1 of 3 strategies: remove all tag-0 atoms,
aggregate all tag-0 atoms into a single super-node, or aggregate all tag-0 atoms of
a given element into a single super-node (hence, up to 3 super nodes will be created
since 0C20 catalysts can have up to 3 elements).

.. image:: https://raw.githubusercontent.com/vict0rsch/phast/main/examples/data/rewiring.png
    :alt: graph rewiring
    :width: 600px

.. code-block:: python

    from phast.graph_rewiring import remove_tag0_nodes

    data = load_oc20_data_batch() # Yours to define
    rewired_data = remove_tag0_nodes(data)

.. warning::
    This modules expects ``torch_geometric`` to be installed.

"""

from copy import deepcopy

import torch
from torch import cat, isin, tensor, where
from phast.utils import ensure_pyg_ok

from typing import Union

try:
    from torch_geometric.utils import coalesce, remove_self_loops, sort_edge_index
    from torch_geometric.data import Batch, Data

except ImportError:
    pass


@ensure_pyg_ok
def remove_tag0_nodes(data: Union[Batch, Data]) -> Union[Batch, Data]:
    """
    Delete sub-surface (``data.tag == 0``) nodes and rewire the graph accordingly.

    .. warning::
        This function modifies the input data in-place.

    Expected ``data`` tensor attributes:
        - ``pos``: node positions
        - ``atomic_numbers``: atomic numbers
        - ``batch``: mini-batch id for each atom
        - ``tags``: atom tags
        - ``edge_index``: edge indices as a $2 \times E$ tensor
        - ``force``: force vectors per atom (optional)
        - ``pos_relaxed``: relaxed atom positions (optional)
        - ``fixed``: mask for fixed atoms (optional)
        - ``natoms``: number of atoms per graph
        - ``ptr``: cumulative sum of ``natoms``
        - ``cell_offsets``: unit cell directional offset for each edge
        - ``distances``: distance between each edge's atoms

    Args:
        data (torch_geometric.Data): the data batch to re-wire
    """
    device = data.edge_index.device

    # non sub-surface atoms
    non_sub = where(data.tags != 0)[0]
    src_is_not_sub = isin(data.edge_index[0], non_sub)
    target_is_not_sub = isin(data.edge_index[1], non_sub)
    neither_is_sub = src_is_not_sub * target_is_not_sub

    # per-atom tensors
    data.pos = data.pos[non_sub, :]
    data.atomic_numbers = data.atomic_numbers[non_sub]
    data.batch = data.batch[non_sub]
    if hasattr(data, "force"):
        data.force = data.force[non_sub, :]
    if hasattr(data, "fixed"):
        data.fixed = data.fixed[non_sub]
    data.tags = data.tags[non_sub]
    if hasattr(data, "pos_relaxed"):
        data.pos_relaxed = data.pos_relaxed[non_sub, :]

    # per-edge tensors
    data.edge_index = data.edge_index[:, neither_is_sub]
    data.cell_offsets = data.cell_offsets[neither_is_sub, :]
    data.distances = data.distances[neither_is_sub]
    # re-index adj matrix, given some nodes were deleted
    num_nodes = data.natoms.sum().item()
    mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    mask[non_sub] = 1
    assoc = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    assoc[mask] = torch.arange(mask.sum(), device=device)
    data.edge_index = assoc[data.edge_index]

    # per-graph tensors
    batch_size = max(data.batch).item() + 1
    data.natoms = tensor(
        [(data.batch == i).sum() for i in range(batch_size)],
        dtype=data.natoms.dtype,
        device=device,
    )
    data.ptr = tensor(
        [0] + [data.natoms[:i].sum() for i in range(1, batch_size + 1)],
        dtype=data.ptr.dtype,
        device=device,
    )
    _, data.neighbors = torch.unique(
        data.batch[data.edge_index[0, :]], return_counts=True
    )

    return data


@ensure_pyg_ok
def one_supernode_per_graph(
    data: Union[Batch, Data], cutoff: float = 6.0, num_elements: int = 83
) -> Union[Batch, Data]:
    """
    Replaces all tag-0 atom with a single super-node $S$ representing them, per graph.
    For each graph, $S$ is the last node in the graph.
    $S$ is positioned at the center of mass of all tag-0 atoms in $x$ and $y$ directions
    but at the maximum $z$ coordinate of all tag-0 atoms.
    All atoms previously connected to a tag-0 atom are now connected to $S$ unless
    that would create an edge longer than ``cutoff``.

    Expected ``data`` attributes are the same as for :func:`remove_tag0_nodes`.

    .. note::
        $S$ will be created with a new atomic number $Z_{S} = num\_elements + 1$,
        so this should be set to the number of elements expected to be present in the
        dataset, not that of the current graph.

    .. warning::
        This function modifies the input data in-place.

    Args:
        data (data.Data): single batch of graphs
    """
    batch_size = max(data.batch).item() + 1
    device = data.edge_index.device
    original_ptr = deepcopy(data.ptr)

    # ids of sub-surface nodes, per batch
    sub_nodes = [
        where((data.tags == 0) * (data.batch == i))[0] for i in range(batch_size)
    ]

    # idem for non-sub-surface nodes
    non_sub_nodes = [
        where((data.tags != 0) * (data.batch == i))[0] for i in range(batch_size)
    ]

    # super node index per batch: they are last in their batch
    # (after removal of tag0 nodes)
    new_sn_ids = [
        sum([len(nsn) for nsn in non_sub_nodes[: i + 1]]) + i for i in range(batch_size)
    ]
    # define new number of atoms per batch
    data.ptr = tensor(
        [0] + [nsi + 1 for nsi in new_sn_ids], dtype=data.ptr.dtype, device=device
    )
    data.natoms = data.ptr[1:] - data.ptr[:-1]
    # Store number of nodes each supernode contains
    data.subnodes = tensor(
        [len(sub) for sub in sub_nodes], dtype=torch.long, device=device
    )

    # super node position for a batch is the mean of its aggregates
    # sn_pos = [data.pos[sub_nodes[i]].mean(0) for i in range(batch_size)]
    sn_pos = [
        cat(
            [
                data.pos[sub_nodes[i], :2].mean(0),
                data.pos[sub_nodes[i], 2].max().unsqueeze(0),
            ],
            dim=0,
        )
        for i in range(batch_size)
    ]
    # the super node force is the mean of the force applied to its aggregates
    if hasattr(data, "force"):
        sn_force = [data.force[sub_nodes[i]].mean(0) for i in range(batch_size)]
        data.force = cat(
            [
                cat([data.force[non_sub_nodes[i]], sn_force[i][None, :]])
                for i in range(batch_size)
            ]
        )

    # learn a new embedding to each supernode
    data.atomic_numbers = cat(
        [
            cat(
                [
                    data.atomic_numbers[non_sub_nodes[i]],
                    tensor([num_elements + 1], device=device),
                ]
            )
            for i in range(batch_size)
        ]
    )

    # position excludes sub-surface atoms but includes extra super-nodes
    data.pos = cat(
        [
            cat([data.pos[non_sub_nodes[i]], sn_pos[i][None, :]])
            for i in range(batch_size)
        ]
    )
    # relaxed position for supernode is the same as initial position
    if hasattr(data, "pos_relaxed"):
        data.pos_relaxed = cat(
            [
                cat([data.pos_relaxed[non_sub_nodes[i]], sn_pos[i][None, :]])
                for i in range(batch_size)
            ]
        )

    # idem, sn position is fixed
    if hasattr(data, "fixed"):
        data.fixed = cat(
            [
                cat(
                    [
                        data.fixed[non_sub_nodes[i]],
                        tensor([1.0], dtype=data.fixed.dtype, device=device),
                    ]
                )
                for i in range(batch_size)
            ]
        )
    # idem, sn have tag0
    data.tags = cat(
        [
            cat(
                [
                    data.tags[non_sub_nodes[i]],
                    tensor([0], dtype=data.tags.dtype, device=device),
                ]
            )
            for i in range(batch_size)
        ]
    )

    # Edge-index and cell_offsets
    batch_idx_adj = data.batch[data.edge_index][0]
    ei_sn = data.edge_index.clone()
    new_cell_offsets = data.cell_offsets.clone()
    # number of nodes in this batch: all existing + batch_size supernodes
    num_nodes = original_ptr[-1].item()
    # Re-index
    mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    mask[cat(non_sub_nodes)] = 1  # mask is 0 for sub-surface atoms
    assoc = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    assoc[mask] = cat(
        [
            torch.arange(data.ptr[e], data.ptr[e + 1] - 1, device=device)
            for e in range(batch_size)
        ]
    )
    # re-index only edges for which not both nodes are sub-surface atoms
    ei_sn = assoc[ei_sn]

    # Adapt cell_offsets: add [0,0,0] for supernode related edges
    is_minus_one = isin(ei_sn, tensor(-1, device=device))
    new_cell_offsets[is_minus_one.any(dim=0)] = tensor([0, 0, 0], device=device)
    # Replace index -1 by supernode index
    ei_sn = where(
        is_minus_one,
        tensor(new_sn_ids, device=device)[batch_idx_adj],
        ei_sn,
    )
    # Remove self loops
    ei_sn, new_cell_offsets = remove_self_loops(ei_sn, new_cell_offsets)

    # Remove tag0 related duplicates
    # First, store tag 1/2 adjacency
    new_non_sub_nodes = where(data.tags != 0)[0]
    tag12_ei = ei_sn[:, isin(ei_sn, new_non_sub_nodes).all(dim=0)]
    tag12_cell_offsets_ei = new_cell_offsets[
        isin(ei_sn, new_non_sub_nodes).all(dim=0), :
    ]
    # Remove duplicate in supernode adjacency
    indxes = isin(ei_sn, tensor(new_sn_ids).to(device=ei_sn.device)).any(dim=0)
    ei_sn, new_cell_offsets = coalesce(
        ei_sn[:, indxes], edge_attr=new_cell_offsets[indxes, :], reduce="min"
    )
    # Merge back both
    ei_sn = cat([tag12_ei, ei_sn], dim=1)
    new_cell_offsets = cat([tag12_cell_offsets_ei, new_cell_offsets], dim=0)
    ei_sn, new_cell_offsets = sort_edge_index(ei_sn, edge_attr=new_cell_offsets)

    # Remove duplicate entries
    # ei_sn, new_cell_offsets = coalesce(
    #   ei_sn, edge_attr=new_cell_offsets, reduce="min",
    # )

    # ensure correct type
    data.edge_index = ei_sn.to(dtype=data.edge_index.dtype)
    data.cell_offsets = new_cell_offsets.to(dtype=data.cell_offsets.dtype)

    # distances
    data.distances = torch.sqrt(
        ((data.pos[data.edge_index[0, :]] - data.pos[data.edge_index[1, :]]) ** 2).sum(
            -1
        )
    ).to(dtype=data.distances.dtype)

    # batch
    data.batch = torch.zeros(data.ptr[-1], dtype=data.batch.dtype, device=device)
    for i, p in enumerate(data.ptr[:-1]):
        data.batch[
            torch.arange(p, data.ptr[i + 1], dtype=torch.long, device=device)
        ] = tensor(i, dtype=data.batch.dtype, device=device)

    return adjust_cutoff_distances(data, new_sn_ids, cutoff)


@ensure_pyg_ok
def one_supernode_per_atom_type(
    data: Union[Batch, Data], cutoff: float = 6.0
) -> Union[Batch, Data]:
    """
    For each graph independently, replace all tag-0 atoms of a given element by a new
    super node $S_i, \ i \in \{1..3\}$. As per :func:`one_supernode_per_graph`, each
    $S_i$ is positioned at the center of mass of the atoms it replaces in $x$ and $y$
    dimensions but at the maximum height of the atoms it replaces in the $z$ dimension.

    Expected ``data`` attributes are the same as for :func:`remove_tag0_nodes`.

    .. note::
        $S_i$ conserves the atomic number of the tag-0 atoms it replaces.

    .. warning::
        This function modifies the input data in-place.

    Args:
        data (torch_geometric.Data): the data batch to re-wire

    Returns:
        torch_geometric.Data: the data rewired data batch
    """
    batch_size = max(data.batch).item() + 1
    device = data.edge_index.device
    original_ptr = deepcopy(data.ptr)

    # idem for non-sub-surface nodes
    non_sub_nodes = [
        where((data.tags != 0) * (data.batch == i))[0] for i in range(batch_size)
    ]
    # atom types per supernode
    atom_types = [
        torch.unique(data.atomic_numbers[(data.tags == 0) * (data.batch == i)])
        for i in range(batch_size)
    ]
    # number of supernodes per batch
    num_supernodes = [atom_types[i].shape[0] for i in range(batch_size)]
    total_num_supernodes = sum(num_supernodes)
    # indexes of nodes belonging to each supernode
    supernodes_composition = [
        where((data.atomic_numbers == an) * (data.tags == 0) * (data.batch == i))[0]
        for i in range(batch_size)
        for an in atom_types[i]
    ]
    # Store number of nodes each supernode regroups
    data.subnodes = tensor(
        [len(sub) for sub in supernodes_composition], dtype=torch.long, device=device
    )

    # super node index per batch: they are last in their batch
    # (after removal of tag0 nodes)
    new_sn_ids = [
        [
            sum([len(nsn) for nsn in non_sub_nodes[: i + 1]]) + j
            for j in range(sum(num_supernodes[:i]), sum(num_supernodes[: i + 1]))
        ]
        for i in range(batch_size)
    ]
    # Concat version
    new_sn_ids_cat = [s for sn in new_sn_ids for s in sn]

    # supernode positions
    supernodes_pos = [
        cat([data.pos[sn, :2].mean(0), data.pos[sn, 2].max().unsqueeze(0)], dim=0)[
            None, :
        ]
        for sn in supernodes_composition
    ]

    # number of atoms per graph in the batch
    data.ptr = tensor(
        [0] + [max(nsi) + 1 for nsi in new_sn_ids],
        dtype=data.ptr.dtype,
        device=device,
    )
    data.natoms = data.ptr[1:] - data.ptr[:-1]

    # batch
    data.batch = cat(
        [
            tensor(i, device=device).expand(
                non_sub_nodes[i].shape[0] + num_supernodes[i]
            )
            for i in range(batch_size)
        ]
    )

    # tags
    data.tags = cat(
        [
            cat(
                [
                    data.tags[non_sub_nodes[i]],
                    tensor([0], dtype=data.tags.dtype, device=device).expand(
                        num_supernodes[i]
                    ),
                ]
            )
            for i in range(batch_size)
        ]
    )

    # re-index edges
    num_nodes = original_ptr[-1]  # + sum(num_supernodes)
    mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    mask[cat(non_sub_nodes)] = 1  # mask is 0 for sub-surface atoms
    assoc = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    assoc[mask] = cat(
        [
            torch.arange(
                data.ptr[e], data.ptr[e + 1] - num_supernodes[e], device=device
            )
            for e in range(batch_size)
        ]
    )
    # Set corresponding supernode index to subatoms
    for i, sn in enumerate(supernodes_composition):
        assoc[sn] = new_sn_ids_cat[i]
    # Re-index
    data.edge_index = assoc[data.edge_index]

    # Adapt cell_offsets: add [0,0,0] for supernode related edges
    data.cell_offsets[
        isin(data.edge_index, tensor(new_sn_ids_cat, device=device)).any(dim=0)
    ] = tensor([0, 0, 0], device=device)

    # Remove self loops and duplicates
    data.edge_index, data.cell_offsets = remove_self_loops(
        data.edge_index, data.cell_offsets
    )

    # Remove tag0 related duplicates
    # First, store tag 1/2 adjacency
    new_non_sub_nodes = where(data.tags != 0)[0]
    tag12_ei = data.edge_index[:, isin(data.edge_index, new_non_sub_nodes).all(dim=0)]
    tag12_cell_offsets_ei = data.cell_offsets[
        isin(data.edge_index, new_non_sub_nodes).all(dim=0), :
    ]
    # Remove duplicate in supernode adjacency
    indxes = isin(
        data.edge_index, tensor(new_sn_ids_cat).to(device=data.edge_index.device)
    ).any(dim=0)
    data.edge_index, data.cell_offsets = coalesce(
        data.edge_index[:, indxes], edge_attr=data.cell_offsets[indxes, :], reduce="min"
    )
    # Merge back both
    data.edge_index = cat([tag12_ei, data.edge_index], dim=1)
    data.cell_offsets = cat([tag12_cell_offsets_ei, data.cell_offsets], dim=0)
    data.edge_index, data.cell_offsets = sort_edge_index(
        data.edge_index, edge_attr=data.cell_offsets
    )

    # SNs are last in their batch
    data.atomic_numbers = cat(
        [
            cat([data.atomic_numbers[non_sub_nodes[i]], atom_types[i]])
            for i in range(batch_size)
        ]
    )

    # position exclude the sub-surface atoms but include extra super-nodes
    acc_num_supernodes = [0] + [sum(num_supernodes[: i + 1]) for i in range(batch_size)]
    data.pos = cat(
        [
            cat(
                [
                    data.pos[non_sub_nodes[i]],
                    cat(
                        supernodes_pos[
                            acc_num_supernodes[i] : acc_num_supernodes[i + 1]
                        ]
                    ),
                ]
            )
            for i in range(batch_size)
        ]
    )

    # pos relaxed
    if hasattr(data, "pos_relaxed"):
        data.pos_relaxed = cat(
            [
                cat(
                    [
                        data.pos_relaxed[non_sub_nodes[i]],
                        cat(
                            supernodes_pos[
                                acc_num_supernodes[i] : acc_num_supernodes[i + 1]
                            ]
                        ),
                    ]
                )
                for i in range(batch_size)
            ]
        )

    # the force applied on the super node is the mean of the force applied
    # to its aggregates (per batch)
    if hasattr(data, "force"):
        sn_force = [
            data.force[supernodes_composition[i]].mean(0)[None, :]
            for i in range(total_num_supernodes)
        ]
        data.force = cat(
            [
                cat(
                    [
                        data.force[non_sub_nodes[i]],
                        cat(
                            sn_force[acc_num_supernodes[i] : acc_num_supernodes[i + 1]]
                        ),
                    ]
                )
                for i in range(batch_size)
            ]
        )

    # fixed atoms
    if hasattr(data, "fixed"):
        data.fixed = cat(
            [
                cat(
                    [
                        data.fixed[non_sub_nodes[i]],
                        tensor([1.0], dtype=data.fixed.dtype, device=device).expand(
                            num_supernodes[i]
                        ),
                    ]
                )
                for i in range(batch_size)
            ]
        )

    # distances
    # TODO: compute with cell_offsets
    data.distances = torch.sqrt(
        ((data.pos[data.edge_index[0, :]] - data.pos[data.edge_index[1, :]]) ** 2).sum(
            -1
        )
    )

    return adjust_cutoff_distances(data, new_sn_ids_cat, cutoff)


@ensure_pyg_ok
def adjust_cutoff_distances(
    data: Union[Data, Batch], sn_indxes: torch.Tensor, cutoff: float = 6.0
) -> Union[Data, Batch]:
    """
    Because of rewiring, some edges could be now longer than
    the allowed cutoff distance. This function removes them.

    Modified attributes:
    * ``edge_index``
    * ``cell_offsets``
    * ``distances``
    * ``neighbors``

    .. warning::
        This function modifies the input data in-place.

    Args:
        data (torch_geometric.Data): The rewired graph data.
        sn_indxes (torch.Tensor[torch.Long]): Indices of the supernodes.
        cutoff (float, optional): Maximum edge length. Defaults to 6.0.

    Returns:
        torch_geometric.Data: The updated graph.
    """
    # remove long edges (> cutoff), for sn related edges only
    sn_indxes = isin(
        data.edge_index, tensor(sn_indxes, device=data.edge_index.device)
    ).any(dim=0)
    cutoff_mask = torch.logical_not((data.distances > cutoff) * sn_indxes)
    data.edge_index = data.edge_index[:, cutoff_mask]
    data.cell_offsets = data.cell_offsets[cutoff_mask, :]
    data.distances = data.distances[cutoff_mask]
    _, data.neighbors = torch.unique(
        data.batch[data.edge_index[0, :]], return_counts=True
    )
    return data
