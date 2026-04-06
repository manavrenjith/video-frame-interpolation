import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity

from src.model import VFIUNet


def _find_triplets(test_dir: Path) -> List[Tuple[Path, Path, Path, str]]:
	triplets: List[Tuple[Path, Path, Path, str]] = []
	for triplet_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
		a_path = triplet_dir / "A.png"
		b_path = triplet_dir / "B.png"
		c_path = triplet_dir / "C.png"
		if not (a_path.exists() and b_path.exists() and c_path.exists()):
			continue

		video_name = triplet_dir.name.split("_triplet_")[0]
		triplets.append((a_path, b_path, c_path, video_name))

	if not triplets:
		raise ValueError(f"No valid triplets found in: {test_dir}")
	return triplets


def _load_rgb_image(path: Path) -> np.ndarray:
	image = cv2.imread(str(path), cv2.IMREAD_COLOR)
	if image is None:
		raise ValueError(f"Failed to load image: {path}")
	return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _to_model_tensor(image_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
	tensor = torch.from_numpy(np.transpose(image_rgb, (2, 0, 1))).float()
	tensor = tensor / 127.5 - 1.0
	return tensor.unsqueeze(0).to(device)


def _to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
	arr = tensor.detach().cpu().squeeze(0).numpy()
	arr = np.transpose(arr, (1, 2, 0))
	arr = np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)
	return arr


def _compute_psnr(pred_rgb: np.ndarray, gt_rgb: np.ndarray) -> float:
	pred_f = pred_rgb.astype(np.float32)
	gt_f = gt_rgb.astype(np.float32)
	mse = np.mean((pred_f - gt_f) ** 2)
	if mse <= 1e-10:
		return 100.0
	return float(20.0 * np.log10(255.0) - 10.0 * np.log10(mse))


def _compute_ssim(pred_rgb: np.ndarray, gt_rgb: np.ndarray) -> float:
	return float(
		structural_similarity(
			gt_rgb,
			pred_rgb,
			channel_axis=2,
			data_range=255,
		)
	)


def _make_comparison(gt_rgb: np.ndarray, pred_rgb: np.ndarray) -> np.ndarray:
	diff = cv2.absdiff(gt_rgb, pred_rgb)
	diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
	diff_color = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
	diff_color = cv2.cvtColor(diff_color, cv2.COLOR_BGR2RGB)
	return np.concatenate([gt_rgb, pred_rgb, diff_color], axis=1)


def _load_model(model_path: Path, device: torch.device) -> VFIUNet:
	model = VFIUNet().to(device)
	checkpoint = torch.load(model_path, map_location=device)

	if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
		state_dict = checkpoint["model_state_dict"]
	elif isinstance(checkpoint, dict):
		state_dict = checkpoint
	else:
		raise ValueError("Unsupported checkpoint format.")

	model.load_state_dict(state_dict)
	model.eval()
	return model


def evaluate(model_path: Path, test_dir: Path, output_dir: Path, device: str) -> None:
	if not model_path.exists():
		raise FileNotFoundError(f"Model file not found: {model_path}")
	if not test_dir.exists():
		raise FileNotFoundError(f"Test directory not found: {test_dir}")

	output_dir.mkdir(parents=True, exist_ok=True)
	device_obj = torch.device(device)
	model = _load_model(model_path, device_obj)
	triplets = _find_triplets(test_dir)

	metrics_by_video: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
	all_psnr: List[float] = []
	all_ssim: List[float] = []

	for index, (a_path, b_path, c_path, video_name) in enumerate(triplets, start=1):
		frame_a = _load_rgb_image(a_path)
		frame_b_gt = _load_rgb_image(b_path)
		frame_c = _load_rgb_image(c_path)

		a_tensor = _to_model_tensor(frame_a, device_obj)
		c_tensor = _to_model_tensor(frame_c, device_obj)
		inputs = torch.cat([a_tensor, c_tensor], dim=1)

		with torch.no_grad():
			pred_b_tensor, _ = model(inputs)

		pred_b_rgb = _to_uint8_image(pred_b_tensor)
		psnr = _compute_psnr(pred_b_rgb, frame_b_gt)
		ssim = _compute_ssim(pred_b_rgb, frame_b_gt)

		all_psnr.append(psnr)
		all_ssim.append(ssim)
		metrics_by_video[video_name].append((psnr, ssim))

		comparison = _make_comparison(frame_b_gt, pred_b_rgb)
		comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
		out_path = output_dir / f"comparison_{index:03d}.png"
		cv2.imwrite(str(out_path), comparison_bgr)

	print("\nEvaluation Summary")
	print("=" * 72)
	print(f"{'Video':<24} {'Avg PSNR (dB)':>16} {'Avg SSIM':>14}")
	print("-" * 72)

	for video_name in sorted(metrics_by_video.keys()):
		video_psnr = [x[0] for x in metrics_by_video[video_name]]
		video_ssim = [x[1] for x in metrics_by_video[video_name]]
		avg_psnr = float(np.mean(video_psnr))
		avg_ssim = float(np.mean(video_ssim))
		print(f"{video_name:<24} {avg_psnr:>16.4f} {avg_ssim:>14.4f}")

	overall_psnr = float(np.mean(all_psnr)) if all_psnr else 0.0
	overall_ssim = float(np.mean(all_ssim)) if all_ssim else 0.0
	status = "PASS" if overall_psnr > 30.0 else "FAIL"

	print("-" * 72)
	print(f"{'OVERALL':<24} {overall_psnr:>16.4f} {overall_ssim:>14.4f}")
	print(f"Result: {status} (threshold: average PSNR > 30.0 dB)")

	if status == "FAIL":
		print(
			"Model did not meet quality threshold. Consider training for more epochs "
			"or adjusting the loss weights in src/train.py"
		)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate trained VFI model on triplet frames.")
	parser.add_argument(
		"--model_path",
		type=Path,
		default=Path("models/best.pth"),
		help="Path to model checkpoint (.pth)",
	)
	parser.add_argument(
		"--test_dir",
		type=Path,
		required=True,
		help="Path to directory containing test triplet folders",
	)
	parser.add_argument(
		"--output_dir",
		type=Path,
		default=Path("evaluation_output"),
		help="Directory to save comparison images",
	)
	parser.add_argument(
		"--device",
		type=str,
		choices=["cpu", "cuda"],
		default="cpu",
		help="Device for inference",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	evaluate(args.model_path, args.test_dir, args.output_dir, args.device)
