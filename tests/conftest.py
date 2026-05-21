"""Pytest configuration and fixtures for Alberta Framework tests."""

import sys
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_path in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


@pytest.fixture
def rng_key():
    """Provide a deterministic JAX random key."""
    return jr.key(42)


@pytest.fixture
def feature_dim():
    """Default feature dimension for tests."""
    return 10


@pytest.fixture
def sample_observation(feature_dim, rng_key):
    """Generate a sample observation vector."""
    return jr.normal(rng_key, (feature_dim,), dtype=jnp.float32)


@pytest.fixture
def sample_target():
    """Generate a sample target value."""
    return jnp.array([1.5], dtype=jnp.float32)
