import random
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


PathLike = Union[str, Path]


class TripletFrameDataset(Dataset):
	"""Dataset of frame triplets (A, B, C) where B is the intermediate ground truth."""

	def __init__(self, root_dir: PathLike, training: bool = True, crop_size: int = 256) -> None:
		self.root_dir = Path(root_dir)
		self.training = training
		self.crop_size = int(crop_size)

		if not self.root_dir.exists():
			raise FileNotFoundError(f"Triplet root directory does not exist: {self.root_dir}")

		self.samples = self._discover_samples()

	def _discover_samples(self) -> List[Tuple[Path, Path, Path]]:
		samples: List[Tuple[Path, Path, Path]] = []

		for triplet_dir in sorted(p for p in self.root_dir.iterdir() if p.is_dir()):
			a_path = triplet_dir / "A.png"
			b_path = triplet_dir / "B.png"
			c_path = triplet_dir / "C.png"

			if a_path.exists() and b_path.exists() and c_path.exists():
				samples.append((a_path, b_path, c_path))
				continue

			png_files = sorted(triplet_dir.glob("*.png"))
			if len(png_files) == 3:
				samples.append((png_files[0], png_files[1], png_files[2]))

		if not samples:
			raise ValueError(f"No valid triplet subfolders found in: {self.root_dir}")

		return samples

	def __len__(self) -> int:
		return len(self.samples)

	def _load_image(self, path: Path) -> np.ndarray:
		image = cv2.imread(str(path), cv2.IMREAD_COLOR)
		if image is None:
			raise ValueError(f"Failed to load image: {path}")
		return image

	def _resize_if_needed(self, images: Sequence[np.ndarray]) -> List[np.ndarray]:
		h, w = images[0].shape[:2]
		if h >= self.crop_size and w >= self.crop_size:
			return list(images)

		target_h = max(h, self.crop_size)
		target_w = max(w, self.crop_size)
		return [cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR) for img in images]

	def _random_crop(self, images: Sequence[np.ndarray]) -> List[np.ndarray]:
		images = self._resize_if_needed(images)
		h, w = images[0].shape[:2]

		y = random.randint(0, h - self.crop_size)
		x = random.randint(0, w - self.crop_size)
		return [img[y : y + self.crop_size, x : x + self.crop_size] for img in images]

	def _center_crop_or_resize(self, images: Sequence[np.ndarray]) -> List[np.ndarray]:
		images = self._resize_if_needed(images)
		h, w = images[0].shape[:2]

		y = (h - self.crop_size) // 2
		x = (w - self.crop_size) // 2
		return [img[y : y + self.crop_size, x : x + self.crop_size] for img in images]

	def _apply_training_augmentations(self, images: Sequence[np.ndarray]) -> List[np.ndarray]:
		aug_images = list(images)

		if random.random() < 0.5:
			aug_images = [cv2.flip(img, 1) for img in aug_images]
		if random.random() < 0.5:
			aug_images = [cv2.flip(img, 0) for img in aug_images]

		aug_images = self._random_crop(aug_images)

		brightness = 1.0 + random.uniform(-0.2, 0.2)
		contrast = 1.0 + random.uniform(-0.2, 0.2)

		jittered = []
		for img in aug_images:
			img_f = img.astype(np.float32)
			img_f = (img_f - 127.5) * contrast + 127.5
			img_f = img_f * brightness
			jittered.append(np.clip(img_f, 0, 255).astype(np.uint8))

		return jittered

	def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).float()
		return tensor / 127.5 - 1.0

	def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		a_path, b_path, c_path = self.samples[index]

		frames = [self._load_image(a_path), self._load_image(b_path), self._load_image(c_path)]

		if self.training:
			frames = self._apply_training_augmentations(frames)
		else:
			frames = self._center_crop_or_resize(frames)

		frame_a, frame_b, frame_c = (self._to_tensor(frame) for frame in frames)
		return frame_a, frame_b, frame_c
