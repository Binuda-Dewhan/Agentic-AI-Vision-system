import os

import pytest


@pytest.fixture(autouse=True)
def setup_env():
    """Setup test environment."""
    os.environ["TESTING"] = "true"
    yield
