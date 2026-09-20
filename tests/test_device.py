"""Which device gets picked, and which dtypes are offered on it.

Both are decisions the server makes for you at startup, on a machine the author of a deployment
may never have seen, so they are worth pinning down. No weights and no accelerator needed: the
availability probes are patched, which is the only way to exercise the CUDA branch from a Mac
and the MPS branch from a Linux box.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("laya")

from engines.laya.loader import Checkpoint, resolve_device, supported_dtypes  # noqa: E402


@pytest.fixture
def no_accelerator(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


def test_an_explicit_device_is_passed_through_untouched(no_accelerator):
    """Including one that is not there: ARBITER_DEVICE=cuda on a Mac should fail, not fall back.

    A silent fall back to the CPU is a tenth of the speed and no message saying so.
    """
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("cuda:1") == "cuda:1"
    assert resolve_device("cpu") == "cpu"


def test_auto_prefers_cuda_then_mps_then_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "mps"

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"


def test_an_empty_setting_counts_as_auto(no_accelerator):
    assert resolve_device("") == "cpu"


def test_ampere_and_up_get_autocast_and_bf16(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: (8, 0))
    assert supported_dtypes(torch.device("cuda")) == ("fp32", "autocast", "bf16")


def test_pre_ampere_gets_neither(monkeypatch):
    """No bf16 tensor cores means bf16 is not a speed-up, so it is not offered as one."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: (7, 5))
    assert supported_dtypes(torch.device("cuda")) == ("fp32",)


def test_mps_gets_whole_model_halves_but_not_autocast():
    """autocast on MPS would be fp16 over a different op set -- one name, two arithmetics."""
    assert supported_dtypes(torch.device("mps")) == ("fp32", "fp16", "bf16")
    assert "autocast" not in supported_dtypes(torch.device("mps"))


def test_cpu_gets_fp32_only():
    assert supported_dtypes(torch.device("cpu")) == ("fp32",)


def test_graphs_mode_is_refused_before_any_weights_are_loaded():
    """The refusal comes first, so a wrong ARBITER_MODE costs a second and not a minute."""
    with pytest.raises(ValueError, match="ARBITER_MODE=eager"):
        Checkpoint("english", "/nowhere/there/are/no/weights", device="cpu", mode="graphs")
