"""Campaign command modules: the local and remote admission drivers, checkpoint pauses, the reconciliation of refused
remote launches and the remote release verification chain.

Most of these modules are scripts: their module-level code IS the command (one driver pass, one pause, one release
step) and acts on the live campaign state at hard-coded paths. Importing such a module runs it. Run them only through
their fixed entry points ``runs/setup/tools/<name>.py`` (same file names). No code in s4d, ops, scripts or tests
imports ``ops.campaign`` (``tests/test_ops_layout.py``); only those entry points refer to it.
"""
