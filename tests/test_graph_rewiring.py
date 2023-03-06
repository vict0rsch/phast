import t_utils as tu
import torch_geometric

from phast import graph_rewiring as pgr


def test_data():
    assert isinstance(tu.get_data(), torch_geometric.data.Batch)


def test_remove_tag0_nodes():
    pgr.remove_tag0_nodes(tu.get_data())


def test_one_supernode_per_graph():
    pgr.one_supernode_per_graph(tu.get_data())


def test_one_supernode_per_atom_type():
    pgr.one_supernode_per_atom_type_new_dist(tu.get_data())


def test_one_supernode_per_atom_type_min_dist():
    pgr.one_supernode_per_atom_type_min_dist(tu.get_data())
