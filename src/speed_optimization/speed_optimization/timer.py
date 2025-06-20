# timer.py
import time
from functools import wraps
from collections import defaultdict


_stats = defaultdict(lambda: {"calls": 0, "elapsed": 0.0})

def timed(fn):
    """Decorator that accumulates wall-clock time (perf_counter) for each call."""
    name = f"{fn.__module__}.{fn.__qualname__}"

    @wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        res = fn(*args, **kwargs)
        _stats[name]["calls"]   += 1
        _stats[name]["elapsed"] += time.perf_counter() - t0
        return res

    return wrapper


def print_stats(min_pct: float = 0.5, sort_key: str = "elapsed"):
    """Pretty-print anything that took ≥ min_pct % of the total runtime."""
    if not _stats:
        print("⚠️  No timing data collected.")
        return

    total = sum(v["elapsed"] for v in _stats.values())
    lines = []
    for name, d in _stats.items():
        pct = 100 * d["elapsed"] / total
        if pct >= min_pct:
            lines.append(
                (d[sort_key], f"{name:60s}  {d['calls']:4d} calls  "
                              f"{d['elapsed']*1e3:9.1f} ms  ({pct:4.1f} %)")
            )
    for _, line in sorted(lines, reverse=True):
        print(line)
    print(f"\nTotal timed runtime: {total:.3f} s")

