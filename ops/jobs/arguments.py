"""Argument inspection shared by job controllers."""

from __future__ import annotations


def option(args: list, flag: str) -> str | None:
    args = list(map(str, args))
    return args[args.index(flag) + 1] if flag in args and args.index(flag) + 1 < len(args) else None

