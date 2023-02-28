try:
    import torch_geometric  # noqa: F401

    PYG_OK = True
except ImportError:
    PYG_OK = False


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
