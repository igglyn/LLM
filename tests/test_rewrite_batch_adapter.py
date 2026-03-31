import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rewrite.patcher_models import EmbeddedPatcherConfig, RewritePatchDecoder, RewritePatchEncoder


def test_encoder_accepts_implicit_sequence_axis():
    cfg = EmbeddedPatcherConfig(d_model=16, encoder_layers=1, decoder_layers=1, n_heads=4)
    encoder = RewritePatchEncoder(patch_size=4, cfg=cfg)

    x = torch.randn(2, 10, 16)
    out = encoder(x)
    assert out.shape == (2, 3, 16)


def test_encoder_accepts_explicit_sequence_axis():
    cfg = EmbeddedPatcherConfig(d_model=16, encoder_layers=1, decoder_layers=1, n_heads=4)
    encoder = RewritePatchEncoder(patch_size=4, cfg=cfg)

    x = torch.randn(2, 3, 10, 16)
    out = encoder(x)
    assert out.shape == (2, 3, 3, 16)


def test_decoder_accepts_explicit_sequence_axis():
    cfg = EmbeddedPatcherConfig(d_model=16, encoder_layers=1, decoder_layers=1, n_heads=4)
    encoder = RewritePatchEncoder(patch_size=4, cfg=cfg)
    decoder = RewritePatchDecoder(patch_size=4, cfg=cfg)

    token_hidden = torch.randn(2, 3, 10, 16)
    patch_hidden = encoder(token_hidden)
    recon = decoder(token_hidden, patch_hidden)

    assert recon.shape == token_hidden.shape


def test_decoder_shape_mismatch_raises():
    cfg = EmbeddedPatcherConfig(d_model=16, encoder_layers=1, decoder_layers=1, n_heads=4)
    decoder = RewritePatchDecoder(patch_size=4, cfg=cfg)

    token_hidden = torch.randn(2, 3, 10, 16)
    bad_patch_hidden = torch.randn(2, 2, 3, 16)

    try:
        decoder(token_hidden, bad_patch_hidden)
        raise AssertionError("Expected ValueError for mismatched batch/seq dims")
    except ValueError as exc:
        assert "batch/seq" in str(exc)
