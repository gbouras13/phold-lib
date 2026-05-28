"""pytest configuration and shared fixtures for pholdlib tests.

Running the test suite
----------------------
# Unit tests only (no model download, no GPU)
pytest tests/

# Integration tests – ProstT5 is downloaded automatically to tests/test_data/model_cache
# if not already cached there (or wherever --model_dir points).
pytest tests/ --run_integration

# Point at a pre-existing cache (skips the download)
pytest tests/ --run_integration --model_dir /path/to/model_cache

# Use the real trained phold CNN weights instead of a random predictor
pytest tests/ --run_integration --checkpoint /path/to/model.pt

# With GPU
pytest tests/ --run_integration --gpu_available

# With multiple CPU threads
pytest tests/ --run_integration --threads 4
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Default model cache used by integration tests when --model_dir is not given.
_DEFAULT_MODEL_DIR = Path(__file__).parent / "test_data" / "model_cache"

# ---------------------------------------------------------------------------
# Ensure that pholdlib.prostt5.model is importable even when the installed
# version of scipy/transformers is broken (e.g. missing linear_sum_assignment).
# Unit tests for CNN / toCPU never invoke T5 internals, so a top-level mock
# for the `transformers` package is sufficient.  Integration tests that actually
# load the model gate themselves behind HAS_TRANSFORMERS.
# ---------------------------------------------------------------------------
try:
    # Access the symbols that model.py needs – this is the actual failure point
    # when scipy is broken, not the bare `import transformers`.
    from transformers import T5EncoderModel, T5Tokenizer  # noqa: F401
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False
    # Replace whatever partial state transformers left in sys.modules with a
    # clean mock so that `from transformers import ...` in model.py returns
    # MagicMock attributes rather than raising.
    sys.modules["transformers"] = MagicMock()


def pytest_addoption(parser):
    parser.addoption(
        "--model_dir",
        action="store",
        default=None,
        help=(
            "Directory where ProstT5 is (or will be) cached. "
            f"Defaults to {_DEFAULT_MODEL_DIR}."
        ),
    )
    parser.addoption(
        "--checkpoint",
        action="store",
        default=None,
        help=(
            "Path to a trained CNN checkpoint (.pt or state-dict .pth). "
            "When omitted, integration tests use a randomly initialised predictor."
        ),
    )
    parser.addoption(
        "--gpu_available",
        action="store_true",
        default=False,
        help="Set if a GPU is available for inference tests.",
    )
    parser.addoption(
        "--threads",
        action="store",
        default=1,
        type=int,
        help="Number of CPU threads to use during inference.",
    )
    parser.addoption(
        "--run_integration",
        action="store_true",
        default=False,
        help="Run integration tests (downloads ProstT5 ~1.6 GB if not cached).",
    )


@pytest.fixture(scope="session")
def model_dir(pytestconfig):
    val = pytestconfig.getoption("model_dir")
    return Path(val) if val else _DEFAULT_MODEL_DIR


@pytest.fixture(scope="session")
def checkpoint(pytestconfig):
    val = pytestconfig.getoption("checkpoint")
    return Path(val) if val else None


@pytest.fixture(scope="session")
def gpu_available(pytestconfig):
    return pytestconfig.getoption("gpu_available")


@pytest.fixture(scope="session")
def threads(pytestconfig):
    return pytestconfig.getoption("threads")


@pytest.fixture(scope="session")
def run_integration(pytestconfig):
    return pytestconfig.getoption("run_integration")
