<p align="center">
<strong><a href="https://github.com/vict0rsch/phast" target="_blank">💻 Code</a></strong>
<strong>&nbsp;&nbsp;•&nbsp;&nbsp;</strong>
<strong><a href="https://phast.readthedocs.io/" target="_blank">Docs 📑</a></strong>
</p>

# PhAST: Physics-Aware, Scalable, and Task-specific GNNs for Accelerated Catalyst Design

Read the [paper](https://openreview.net/forum?id=hHercGKiXvP)

Also: https://github.com/vict0rsch/faenet

## Tests

This requires [`poetry`](https://python-poetry.org/docs/). Make sure to have `torch` and `torch_geometric` installed in your environment before you can run the tests. Unfortunately because of CUDA/torch compatibilities, neither `torch` nor `torch_geometric` are part of the explicit dependencies and must be installed independently.

```bash
git clone git@github.com:vict0rsch/phast.git
poetry install --with dev
pytest --cov=phast --cov-report term-missing
```

Testing on Macs you may encounter a [Library Not Loaded Error](https://github.com/pyg-team/pytorch_geometric/issues/6530)
