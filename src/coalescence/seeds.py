"""
Deterministic random number generator initialization.

Every composition that involves randomness should start with a call to
`init_seed()`. This seeds both Python's `random` module and `numpy.random`
from the same seed, so all stochastic decisions in the composition can be
reproduced by re-running with that seed.

The returned seed is what should be recorded in the artwork's metadata
sidecar. To regenerate a curated piece bit-for-bit, future-you (or anyone
else) calls `init_seed(<seed-from-sidecar>)` before running the sketch.

Usage:

    from coalescence.seeds import init_seed

    seed = init_seed()              # generates a fresh seed
    seed = init_seed(4823)          # uses the given seed
    print(f"Composition seed: {seed}")
"""

from __future__ import annotations

import random
import time

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


def init_seed(seed: int | None = None) -> int:
    """
    Seed all known random number generators.

    Parameters
    ----------
    seed : int or None
        If an int, use that as the seed.
        If None, generate a fresh seed from the system clock.

    Returns
    -------
    int
        The seed that was used. Always store this in your artwork's metadata.

    Notes
    -----
    The seed is constrained to the range [0, 2**31 - 1]. This is well below
    Python's unbounded int range but matches NumPy's legacy seeding contract,
    and it's small enough to be human-readable in filenames and logs.
    """
    if seed is None:
        # Use nanoseconds-since-epoch, masked to a 31-bit positive int.
        # This is more granular than time.time() (which is only millisecond
        # resolution on Windows) and avoids collisions when several
        # compositions are run within the same second.
        seed = time.time_ns() & 0x7FFFFFFF

    if not isinstance(seed, int):
        raise TypeError(f"seed must be int or None, got {type(seed).__name__}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    random.seed(seed)
    if _HAS_NUMPY:
        np.random.seed(seed)

    return seed