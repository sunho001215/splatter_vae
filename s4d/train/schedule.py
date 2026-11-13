"""Learning-rate functions. Legacy cosine arithmetic is deliberately unchanged."""

import math

from s4d.config import get


def lr_function(cfg):
    total = int(get(cfg, "train.steps", 200000))
    warmup = int(get(cfg, "train.warmup_steps", 10000))
    lr = float(get(cfg, "train.lr", 5e-4))
    minimum = float(get(cfg, "train.min_lr", 1e-5))
    mode = get(cfg, "train.lr_schedule", "cosine")
    cycles = get(cfg, "train.restarts", [])
    if mode not in ("cosine", "cosine_restarts"):
        raise ValueError("unknown train.lr_schedule")
    if mode == "cosine" and cycles:
        raise ValueError("train.restarts requires cosine_restarts")
    original_end = total
    if mode == "cosine_restarts":
        original_end = get(cfg, "train.original_steps")
        if type(original_end) is not int or not 0 <= warmup < original_end < total:
            raise ValueError("restarts need warmup < original_steps < steps")
        if not 0 < minimum < lr or not isinstance(cycles, list) or not cycles:
            raise ValueError("restarts need 0 < min_lr < lr and nonempty cycles")
        end = original_end
        for c in cycles:
            if (
                not isinstance(c, dict)
                or set(c) != {"start", "end", "warmup", "peak_lr"}
                or any(type(c[k]) is not int for k in ("start", "end", "warmup"))
                or c["start"] != end
                or not 0 < c["warmup"] < c["end"] - c["start"]
                or not math.isfinite(float(c["peak_lr"]))
                or not minimum < float(c["peak_lr"]) <= lr
            ):
                raise ValueError("restart cycles must be contiguous, positive, with min_lr < peak_lr <= lr")
            end = c["end"]
        if end != total:
            raise ValueError("restart cycles must end at train.steps")
    stop = get(cfg, "train.stop_step")
    if mode == "cosine_restarts" and stop is not None and (type(stop) is not int or not 0 < stop <= total):
        raise ValueError("train.stop_step must be in (0, steps]")
    min_ratio = minimum / lr

    def multiplier(step):
        if mode == "cosine_restarts" and step >= original_end:
            for c in cycles:
                if step <= c["end"]:
                    elapsed = step - c["start"]
                    peak = float(c["peak_lr"]) / lr
                    if elapsed <= c["warmup"]:
                        return min_ratio + (peak - min_ratio) * elapsed / c["warmup"]
                    progress = (elapsed - c["warmup"]) / (c["end"] - c["start"] - c["warmup"])
                    return min_ratio + (peak - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
            return min_ratio
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, original_end - warmup))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return multiplier


def training_end(cfg):
    lr_function(cfg)  # validate even when this job has already reached its endpoint
    total = int(get(cfg, "train.steps", 200000))
    stop = get(cfg, "train.stop_step")
    return min(total, int(stop)) if stop else total
