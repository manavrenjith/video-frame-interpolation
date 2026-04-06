import torch

from src.model import VFIUNet


def test_forward_pass_batch_size_1() -> None:
	model = VFIUNet()
	x = torch.randn(1, 6, 256, 256)

	frame, confidence = model(x)

	assert frame.shape == (1, 3, 256, 256)
	assert confidence.shape == (1, 1, 256, 256)


def test_forward_pass_batch_size_4() -> None:
	model = VFIUNet()
	x = torch.randn(4, 6, 256, 256)

	frame, confidence = model(x)

	assert frame.shape == (4, 3, 256, 256)
	assert confidence.shape == (4, 1, 256, 256)


def test_confidence_map_range() -> None:
	model = VFIUNet()
	x = torch.randn(2, 6, 256, 256)

	_, confidence = model(x)

	assert torch.all(confidence >= 0.0)
	assert torch.all(confidence <= 1.0)


def test_model_save_and_load(tmp_path) -> None:
	model = VFIUNet().eval()
	x = torch.randn(1, 6, 256, 256)

	with torch.no_grad():
		frame_before, confidence_before = model(x)

	checkpoint_path = tmp_path / "vfi_unet_state_dict.pt"
	torch.save(model.state_dict(), checkpoint_path)

	loaded_model = VFIUNet().eval()
	loaded_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

	with torch.no_grad():
		frame_after, confidence_after = loaded_model(x)

	assert frame_after.shape == (1, 3, 256, 256)
	assert confidence_after.shape == (1, 1, 256, 256)
	assert torch.allclose(frame_before, frame_after, atol=1e-6, rtol=1e-5)
	assert torch.allclose(confidence_before, confidence_after, atol=1e-6, rtol=1e-5)
