# PhAST

Cf https://github.com/vict0rsch/faenet


TODO

* `get_pbc_distances` in super nodes
* fix distances in `one_supernode_per_atom_type_dist`

## Tests

This requires [`poetry`](https://python-poetry.org/docs/). Make sure to have `torch` and `torch_geometric` installed in your environment before you can run the tests. Unfortunately because of CUDA/torch compatibilities, neither `torch` nor `torch_geometric` are part of the explicit dependencies and must be installed independently.

```bash
git clone git@github.com:vict0rsch/phast.git
poetry install --with dev
pytest --cov=phast --cov-report term-missing
```

Testing on Macs you may encounter a [Library Not Loaded Error](https://github.com/pyg-team/pytorch_geometric/issues/6530)
