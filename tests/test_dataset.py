from pathlib import Path

import cv2
import numpy as np
import torch

from src.dataset import TripletFrameDataset


def _write_triplet(triplet_dir: Path, seed: int) -> None:
	rng = np.random.default_rng(seed)
	triplet_dir.mkdir(parents=True, exist_ok=True)

	for name in ["A.png", "B.png", "C.png"]:
		image = rng.integers(0, 256, size=(300, 320, 3), dtype=np.uint8)
		ok = cv2.imwrite(str(triplet_dir / name), image)
		assert ok, f"failed to write test image: {triplet_dir / name}"


def _make_dataset_root(tmp_path: Path) -> Path:
	root = tmp_path / "triplets"
	_write_triplet(root / "video001_triplet_0001", seed=1)
	_write_triplet(root / "video001_triplet_0002", seed=2)
	return root


def test_triplet_dataset_loads_without_errors(tmp_path: Path) -> None:
	root = _make_dataset_root(tmp_path)
	dataset = TripletFrameDataset(root, training=False)
	assert dataset is not None


def test_getitem_returns_correct_shape(tmp_path: Path) -> None:
	root = _make_dataset_root(tmp_path)
	dataset = TripletFrameDataset(root, training=False)

	frame_a, frame_b, frame_c = dataset[0]

	assert isinstance(frame_a, torch.Tensor)
	assert isinstance(frame_b, torch.Tensor)
	assert isinstance(frame_c, torch.Tensor)

	assert frame_a.shape == (3, 256, 256)
	assert frame_b.shape == (3, 256, 256)
	assert frame_c.shape == (3, 256, 256)


def test_getitem_values_are_normalized_to_minus1_1(tmp_path: Path) -> None:
	root = _make_dataset_root(tmp_path)
	dataset = TripletFrameDataset(root, training=False)

	frame_a, frame_b, frame_c = dataset[0]
	for tensor in [frame_a, frame_b, frame_c]:
		assert float(torch.min(tensor)) >= -1.0
		assert float(torch.max(tensor)) <= 1.0


def test_len_returns_correct_count(tmp_path: Path) -> None:
	root = _make_dataset_root(tmp_path)
	dataset = TripletFrameDataset(root, training=False)
	assert len(dataset) == 2
