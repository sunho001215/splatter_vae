"""Load a private copy of a legacy module without mutating its module or globals.

New entrypoints opt into these copies. Existing scripts and placed execution
scopes continue importing the original byte-identical modules.
"""
import importlib.util
import sys


def private_copy(module):
    name = module.__name__ + "_paired"
    spec = importlib.util.spec_from_file_location(name, module.__file__)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result  # dataclasses resolve their defining module here
    spec.loader.exec_module(result)
    return result
