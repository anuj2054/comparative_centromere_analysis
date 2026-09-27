"""CentroSeek: rank candidate alpha-satellite HOR arrays within a chromosome."""
from .features import array_features, WINDOW, STEP

__version__ = "1.0.0"
__all__ = ["array_features", "rank_arrays", "load_model", "WINDOW", "STEP"]


# rank is imported lazily. Importing it here would put it in sys.modules before
# `python -m centroseek.rank` runs, and runpy then warns that it is re-executing a
# module that is already imported. That invocation is the one the README documents,
# so it must not emit a warning. PEP 562 module __getattr__ keeps the public API
# identical: `from centroseek import rank_arrays` still works.
def __getattr__(name):
    if name in ("rank_arrays", "load_model"):
        from . import rank
        return getattr(rank, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
