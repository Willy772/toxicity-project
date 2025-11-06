import os
import pytest

# Skip total en CI
if os.getenv("CI") == "true":
    pytest.skip("offline model test skipped on CI", allow_module_level=True)

def test_offline_local_only():
    # Import TensorFlow uniquement dans le test (et seulement en local)
    import tensorflow as tf  # noqa: F401
    assert True
