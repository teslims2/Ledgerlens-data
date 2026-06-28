"""Pytest configuration and shared fixtures for the LedgerLens test suite."""

import pytest

from tests.test_factories import synthetic_stellar_trades  # noqa: F401


@pytest.fixture(autouse=True)
def reset_random_state():
    """Reset random seed before each test for reproducibility."""
    import random
    import numpy as np
    
    random.seed(42)
    np.random.seed(42)
    yield
