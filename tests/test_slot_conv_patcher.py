import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rewrite.patcher_models import SlotConvAutoencoder, SlotConvDecoder, SlotConvEncoder


def test_slot_conv_encoder_shape():
    enc = SlotConvEncoder(groups=4, d_chunk=8, d_model=32)
    x = torch.randn(2, 5, 4, 8)
    y = enc(x)
    assert y.shape == (2, 5, 32)


def test_slot_conv_decoder_shape():
    dec = SlotConvDecoder(groups=4, d_chunk=8, d_model=32)
    x = torch.randn(2, 5, 32)
    y = dec(x)
    assert y.shape == (2, 5, 4, 8)


def test_slot_conv_autoencoder_roundtrip_shape():
    ae = SlotConvAutoencoder(groups=4, d_chunk=8, d_model=32)
    x = torch.randn(2, 5, 4, 8)
    recon, lat = ae(x)
    assert lat.shape == (2, 5, 32)
    assert recon.shape == x.shape


def test_slot_conv_encoder_bad_shape_raises():
    enc = SlotConvEncoder(groups=4, d_chunk=8, d_model=32)
    bad = torch.randn(2, 5, 3, 8)
    with pytest.raises(ValueError):
        enc(bad)


def test_slot_conv_decoder_bad_last_dim_raises():
    dec = SlotConvDecoder(groups=4, d_chunk=8, d_model=32)
    bad = torch.randn(2, 5, 16)
    with pytest.raises(ValueError):
        dec(bad)
