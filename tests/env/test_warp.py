import pytest

from env.warp import require_warp_memory


def test_warp_refused_while_jax_preallocates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    with pytest.raises(RuntimeError, match="XLA_PYTHON_CLIENT_PREALLOCATE=false"):
        require_warp_memory()


def test_warp_allowed_without_preallocation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    require_warp_memory()
