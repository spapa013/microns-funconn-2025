import pandas as pd
import time
from contextlib import contextmanager
from joblib import Parallel, delayed
import logging


def apply(func, iterable, n_jobs=1, **kwargs):
    """Apply a function to an iterable in parallel using multiprocessing"""
    return Parallel(n_jobs=n_jobs, **kwargs)(delayed(func)(i) for i in iterable)


@contextmanager
def timer(name: str, log_level: str = "info"):
    """
    Context manager to time code execution.

    Parameters:
        name: Name of the operation being timed
        log_level: Logging level to use ('info', 'debug', etc.)
    """
    logger = logging.getLogger(__name__)
    start_time = time.time()
    yield
    elapsed = time.time() - start_time

    log_fn = getattr(logger, log_level.lower())
    log_fn(f"{name} completed in {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
