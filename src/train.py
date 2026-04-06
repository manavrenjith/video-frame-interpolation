import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from src.dataset import TripletFrameDataset
from src.model import VFIUNet


def _sobel_edge_map(images: torch.Tensor) -> torch.Tensor:
	"""Compute per-channel Sobel gradient magnitude maps."""
	kernel_x = torch.tensor(
		[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
		dtype=images.dtype,
		device=images.device,
	)
	kernel_y = torch.tensor(
		[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
		dtype=images.dtype,
		device=images.device,
	)

	channels = images.shape[1]
	kernel_x = kernel_x.unsqueeze(1).repeat(channels, 1, 1, 1)
	kernel_y = kernel_y.unsqueeze(1).repeat(channels, 1, 1, 1)

	grad_x = F.conv2d(images, kernel_x, padding=1, groups=channels)
	grad_y = F.conv2d(images, kernel_y, padding=1, groups=channels)
	return torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-6)


def _combined_loss(
	pred_frame: torch.Tensor,
	conf_map: torch.Tensor,
	frame_a: torch.Tensor,
	frame_c: torch.Tensor,
	gt_frame_b: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
	# 1) Pixel reconstruction term.
	l1_loss = F.l1_loss(pred_frame, gt_frame_b)

	# 2) Edge consistency term via Sobel gradients.
	pred_edges = _sobel_edge_map(pred_frame)
	gt_edges = _sobel_edge_map(gt_frame_b)
	grad_loss = F.l1_loss(pred_edges, gt_edges)

	# 3) Confidence supervision: larger confidence where frame A is closer to target B.
	diff_a = torch.mean(torch.abs(frame_a - gt_frame_b), dim=1, keepdim=True)
	diff_c = torch.mean(torch.abs(frame_c - gt_frame_b), dim=1, keepdim=True)
	weight_a = torch.exp(-diff_a)
	weight_c = torch.exp(-diff_c)
	conf_target = weight_a / (weight_a + weight_c + 1e-6)
	conf_loss = F.l1_loss(conf_map, conf_target.detach())

	total = 1.0 * l1_loss + 0.5 * grad_loss + 0.1 * conf_loss
	metrics = {
		"total": float(total.detach().item()),
		"l1": float(l1_loss.detach().item()),
		"grad": float(grad_loss.detach().item()),
		"conf": float(conf_loss.detach().item()),
	}
	return total, metrics


def _run_epoch(
	model: nn.Module,
	loader: DataLoader,
	device: torch.device,
	optimizer: Adam | None,
) -> float:
	is_train = optimizer is not None
	model.train(mode=is_train)

	total_loss = 0.0
	num_batches = 0

	for frame_a, frame_b, frame_c in loader:
		frame_a = frame_a.to(device, non_blocking=True)
		frame_b = frame_b.to(device, non_blocking=True)
		frame_c = frame_c.to(device, non_blocking=True)

		inputs = torch.cat([frame_a, frame_c], dim=1)

		if is_train:
			optimizer.zero_grad(set_to_none=True)

		with torch.set_grad_enabled(is_train):
			pred_frame, conf_map = model(inputs)
			loss, _ = _combined_loss(pred_frame, conf_map, frame_a, frame_c, frame_b)

			if is_train:
				loss.backward()
				optimizer.step()

		total_loss += float(loss.detach().item())
		num_batches += 1

	if num_batches == 0:
		return 0.0
	return total_loss / num_batches


def train(config: dict) -> None:
	data_dir = config["data_dir"]
	epochs = int(config["epochs"])
	batch_size = int(config["batch_size"])
	learning_rate = float(config["learning_rate"])
	checkpoint_dir = Path(config["checkpoint_dir"])
	device_str = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

	device = torch.device(device_str)
	checkpoint_dir.mkdir(parents=True, exist_ok=True)

	train_full_dataset = TripletFrameDataset(data_dir, training=True)
	val_full_dataset = TripletFrameDataset(data_dir, training=False)
	dataset_len = len(train_full_dataset)
	train_len = max(1, int(dataset_len * 0.9))
	val_len = dataset_len - train_len
	if val_len == 0:
		val_len = 1
		train_len = dataset_len - 1

	generator = torch.Generator().manual_seed(42)
	indices = torch.randperm(dataset_len, generator=generator).tolist()
	train_indices = indices[:train_len]
	val_indices = indices[train_len : train_len + val_len]

	train_dataset = Subset(train_full_dataset, train_indices)
	val_dataset = Subset(val_full_dataset, val_indices)

	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=0,
		pin_memory=(device.type == "cuda"),
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=0,
		pin_memory=(device.type == "cuda"),
	)

	model = VFIUNet().to(device)
	optimizer = Adam(model.parameters(), lr=learning_rate)
	scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

	best_val_loss = float("inf")
	early_stop_patience = 5
	epochs_without_improvement = 0

	for epoch in range(1, epochs + 1):
		epoch_start = time.time()

		train_loss = _run_epoch(model, train_loader, device, optimizer)
		with torch.no_grad():
			val_loss = _run_epoch(model, val_loader, device, optimizer=None)

		scheduler.step(val_loss)
		current_lr = optimizer.param_groups[0]["lr"]
		epoch_time = time.time() - epoch_start

		print(
			f"Epoch {epoch}/{epochs} | "
			f"train_loss: {train_loss:.6f} | "
			f"val_loss: {val_loss:.6f} | "
			f"lr: {current_lr:.6e} | "
			f"time: {epoch_time:.2f}s"
		)

		checkpoint = {
			"epoch": epoch,
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"scheduler_state_dict": scheduler.state_dict(),
			"train_loss": train_loss,
			"val_loss": val_loss,
			"config": config,
		}

		epoch_path = checkpoint_dir / f"epoch_{epoch}.pth"
		torch.save(checkpoint, epoch_path)

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			epochs_without_improvement = 0
			best_path = checkpoint_dir / "best.pth"
			torch.save(checkpoint, best_path)
		else:
			epochs_without_improvement += 1

		if epochs_without_improvement >= early_stop_patience:
			print(
				f"Early stopping at epoch {epoch}: "
				f"validation loss did not improve for {early_stop_patience} epochs."
			)
			break

