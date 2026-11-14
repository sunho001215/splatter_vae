"""Process-local compatibility imports, installed only by new S/SF entrypoints.

Never call this from a scheduler or an existing row. Install before importing
any trainer/store so all its imports agree on the label-aware gate and identity.
"""
import sys
from s4d import rwlearn
from s4d.rwlearn import dsreset_configs, dsreset_gate, dsreset_identity, dsreset_outcomes, dsreset_tasks

from s4d.rwlearn.dsreset_entrypoints import ENTRYPOINTS


def install():
    for name, module in {"tasks": dsreset_tasks, "identity": dsreset_identity, "gate": dsreset_gate._impl,
                         "configs": dsreset_configs._impl, "outcomes": dsreset_outcomes._impl}.items():
        sys.modules["s4d.rwlearn." + name] = module
        setattr(rwlearn, name, module)
