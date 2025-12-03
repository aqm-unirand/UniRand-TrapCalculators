"""
DEPRECATED: use `import fermiqp.lattice` instead.
This shim will be removed in a future release.
"""
import importlib, sys, warnings

_mod = importlib.import_module(__name__.replace("opticallattice", "lattice"))
sys.modules[__name__] = _mod 
sys.modules["opticallattice"] = _mod
warnings.warn("'opticallattice' is deprecated; use 'lattice' instead.",
              DeprecationWarning, stacklevel=2)