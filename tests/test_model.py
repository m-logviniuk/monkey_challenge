"""Forward-pass, parameter-count, and conv/kan head parity tests."""

import torch

from monkey.model import UNetDensity, count_params


def _build(head):
    return UNetDensity(in_ch=3, n_classes=2, c=16, head=head).eval()


def test_forward_shape_and_finite():
    x = torch.randn(2, 3, 64, 64)
    for head in ("conv", "kan"):
        with torch.no_grad():
            out = _build(head)(x)
        assert out.shape == (2, 2, 64, 64)
        assert torch.isfinite(out).all()


def test_conv_kan_head_param_parity():
    conv = count_params(_build("conv"))
    kan = count_params(_build("kan"))
    assert conv["total"] == kan["total"]
    assert conv["head"] == kan["head"]
    assert conv["kan"] == 0
    assert kan["kan"] > 0


def test_unknown_head_rejected():
    import pytest

    with pytest.raises(ValueError):
        UNetDensity(head="mlp")
