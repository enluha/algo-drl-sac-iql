"""
MVP - Process-level parallelism (multiprocessing) with a simple API.
"""

from typing import Iterable, Callable, Any, List

def run_parallel(tasks: Iterable, n_workers: int, func: Callable[[Any], Any]) -> List[Any]:
    """
    Run func over tasks with n_workers processes.
    - If n_workers==1, run sequentially (no pickling overhead).
    - If n_workers>1, use multiprocessing.Pool.map (func must be top-level).
    """
    tasks = list(tasks)
    if n_workers <= 1:
        return [func(t) for t in tasks]

    from multiprocessing import get_context
    ctx = get_context("spawn")  # safe on Windows/macOS/Linux
    with ctx.Pool(processes=n_workers) as pool:
        return pool.map(func, tasks)
