import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rewrite.train_slot_conv_patcher import _resolve_slot_conv_dims, _validate_known_keys


def test_resolve_slot_conv_dims_with_primary_keys():
    groups, d_chunk = _resolve_slot_conv_dims({"groups": 8, "d_chunk": 4}, d_model=32)
    assert (groups, d_chunk) == (8, 4)


def test_resolve_slot_conv_dims_with_alias_keys():
    groups, d_chunk = _resolve_slot_conv_dims({"group_count": 8, "chunk_dim": 4}, d_model=32)
    assert (groups, d_chunk) == (8, 4)


def test_resolve_slot_conv_dims_can_infer_chunk():
    groups, d_chunk = _resolve_slot_conv_dims({"groups": 8}, d_model=32)
    assert (groups, d_chunk) == (8, 4)


def test_validate_known_keys_reports_suggestion():
    with pytest.raises(ValueError) as exc:
        _validate_known_keys("patcher2", {"gruops": 8}, {"groups", "d_chunk"})
    assert "did you mean 'groups'" in str(exc.value)
