<p align="center">
<strong><a href="https://github.com/vict0rsch/phast" target="_blank">💻 Code</a></strong>
<strong>&nbsp;&nbsp;•&nbsp;&nbsp;</strong>
<strong><a href="https://phast.readthedocs.io/" target="_blank">Docs 📑</a></strong>
</p>

# PhAST: Physics-Aware, Scalable, and Task-specific GNNs for Accelerated Catalyst Design

This repository contains implementations for 2 of the PhAST components presented in the [paper](https://arxiv.org/abs/2211.12020):

* `PhysEmbedding` that allows one to create an embedding vector from atomic numbers that is the concatenation of:
  * A learned embedding for the atom's group
  * A learned embedding for the atom's period
  * A fixed or learned embedding from a set of known physical properties, as reported by [`mendeleev`](https://mendeleev.readthedocs.io/en/stable/data.html#elements)
  * In the case of the OC20 dataset, a learned embedding for the atom's tag (adsorbate, catalyst surface or catalyst sub-surface)
* Tag-based **graph rewiring** strategies for the OC20 dataset:
  * `remove_tag0_nodes` deletes all nodes in the graph associated with a tag 0 and recomputes edges
  * `one_supernode_per_graph` replaces all tag 0 atoms with a single new atom
  * `one_supernode_per_atom_typs` replaces all tag 0 atoms *of a given element* with its own super node

Also: https://github.com/vict0rsch/faenet

## Tests

This requires [`poetry`](https://python-poetry.org/docs/). Make sure to have `torch` and `torch_geometric` installed in your environment before you can run the tests. Unfortunately because of CUDA/torch compatibilities, neither `torch` nor `torch_geometric` are part of the explicit dependencies and must be installed independently.

```bash
git clone git@github.com:vict0rsch/phast.git
poetry install --with dev
pytest --cov=phast --cov-report term-missing
```

Testing on Macs you may encounter a [Library Not Loaded Error](https://github.com/pyg-team/pytorch_geometric/issues/6530)
