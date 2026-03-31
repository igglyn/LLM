from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rewrite.dataloader import ExportedMetaPatchDataset, export_second_patcher_windows


def test_export_second_patcher_windows_and_dataset(tmp_path: Path):
    source_dir = tmp_path / "train_sources"
    export_dir = tmp_path / "patcher2_train"
    source_dir.mkdir(parents=True, exist_ok=True)

    # 16 tokens = 8 first-layer patches at patch_size=2.
    np.save(source_dir / "sample.npy", np.arange(16, dtype=np.int32))

    exported = export_second_patcher_windows(
        source_dir=source_dir,
        export_dir=export_dir,
        patch_size=2,
        meta_patch_size=2,
        window_meta_patches=2,
    )

    assert len(exported) == 1
    arr = np.load(exported[0])
    assert arr.shape == (2, 4, 2)
    assert arr.dtype == np.int32
    np.testing.assert_array_equal(arr[0].reshape(-1), np.arange(8, dtype=np.int32))
    np.testing.assert_array_equal(arr[1].reshape(-1), np.arange(8, 16, dtype=np.int32))

    ds = ExportedMetaPatchDataset(exported)
    assert len(ds) == 2
    assert tuple(ds[0].shape) == (4, 2)
