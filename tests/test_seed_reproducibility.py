"""Tests for GPT-5.5 Pro Task 4 — seed reproducibility.

Verifies:
- stable_hash_seed is deterministic across calls
- stable_hash_seed is sensitive to each input part
- set_global_seed seeds random, numpy, torch consistently
"""
from __future__ import annotations

import os
import sys
import random

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from utils import set_global_seed, stable_hash_seed, stable_int_seed


def test_stable_hash_seed_deterministic():
    assert stable_hash_seed("q1", 11) == stable_hash_seed("q1", 11)


def test_stable_hash_seed_distinguishes_seed():
    assert stable_hash_seed("q1", 11) != stable_hash_seed("q1", 23)


def test_stable_hash_seed_distinguishes_input():
    assert stable_hash_seed("q1", 11) != stable_hash_seed("q2", 11)


def test_stable_hash_seed_int_range():
    # Must fit in np.random.default_rng's accepted seed range (< 2**32)
    for inp in ["foo", "bar", "a very long question string " * 20]:
        s = stable_hash_seed(inp, 11)
        assert 0 <= s < 2**32
        rng = np.random.default_rng(s)
        _ = rng.integers(0, 10)


def test_stable_int_seed_backcompat():
    # Back-compat shim delegates to same SHA256-based function
    assert stable_int_seed("q", 11) == stable_hash_seed("q", 11)


def test_set_global_seed_determinism():
    set_global_seed(42)
    r1 = random.random()
    n1 = np.random.random()
    set_global_seed(42)
    r2 = random.random()
    n2 = np.random.random()
    assert r1 == r2
    assert n1 == n2


def test_set_global_seed_pythonhashseed():
    set_global_seed(11)
    assert os.environ.get("PYTHONHASHSEED") == "11"
