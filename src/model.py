import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
	def __init__(self, in_channels: int, out_channels: int) -> None:
		super().__init__()
		self.block = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.LeakyReLU(0.2, inplace=True),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.block(x)


class AttentionGate(nn.Module):
	def __init__(self, gating_channels: int, skip_channels: int, inter_channels: int) -> None:
		super().__init__()
		self.gating_conv = nn.Conv2d(gating_channels, inter_channels, kernel_size=1, bias=True)
		self.skip_conv = nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=True)
		self.psi = nn.Sequential(
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
			nn.Sigmoid(),
		)

	def forward(self, gating: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
		g_proj = self.gating_conv(gating)
		x_proj = self.skip_conv(skip)
		alpha = self.psi(g_proj + x_proj)
		return skip * alpha


class VFIUNet(nn.Module):
	def __init__(self) -> None:
		super().__init__()

		enc_channels = [64, 128, 256, 512, 512]

		self.enc1 = ConvBlock(6, enc_channels[0])
		self.enc2 = ConvBlock(enc_channels[0], enc_channels[1])
		self.enc3 = ConvBlock(enc_channels[1], enc_channels[2])
		self.enc4 = ConvBlock(enc_channels[2], enc_channels[3])
		self.enc5 = ConvBlock(enc_channels[3], enc_channels[4])

		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
		self.bottleneck = ConvBlock(enc_channels[4], 512)

		self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
		self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
		self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
		self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
		self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

		self.att5 = AttentionGate(512, 512, 256)
		self.att4 = AttentionGate(512, 512, 256)
		self.att3 = AttentionGate(256, 256, 128)
		self.att2 = AttentionGate(128, 128, 64)
		self.att1 = AttentionGate(64, 64, 32)

		self.dec5 = ConvBlock(1024, 512)
		self.dec4 = ConvBlock(1024, 512)
		self.dec3 = ConvBlock(512, 256)
		self.dec2 = ConvBlock(256, 128)
		self.dec1 = ConvBlock(128, 64)

		self.output_head = nn.Conv2d(64, 4, kernel_size=1)

		self._initialize_weights()

	@staticmethod
	def _match_spatial(tensor: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
		if tensor.shape[-2:] != ref.shape[-2:]:
			return F.interpolate(tensor, size=ref.shape[-2:], mode="bilinear", align_corners=False)
		return tensor

	def _initialize_weights(self) -> None:
		for module in self.modules():
			if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
				nn.init.kaiming_normal_(module.weight, a=0.2, mode="fan_out", nonlinearity="leaky_relu")
				if module.bias is not None:
					nn.init.zeros_(module.bias)
			elif isinstance(module, nn.BatchNorm2d):
				nn.init.ones_(module.weight)
				nn.init.zeros_(module.bias)

	@staticmethod
	def _forward_warp(frame_a: torch.Tensor) -> torch.Tensor:
		return frame_a

	@staticmethod
	def _backward_warp(frame_b: torch.Tensor) -> torch.Tensor:
		return frame_b

	def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		e1 = self.enc1(x)
		e2 = self.enc2(self.pool(e1))
		e3 = self.enc3(self.pool(e2))
		e4 = self.enc4(self.pool(e3))
		e5 = self.enc5(self.pool(e4))

		b = self.bottleneck(self.pool(e5))

		d5 = self.up5(b)
		d5 = self._match_spatial(d5, e5)
		s5 = self.att5(d5, e5)
		d5 = self.dec5(torch.cat([d5, s5], dim=1))

		d4 = self.up4(d5)
		d4 = self._match_spatial(d4, e4)
		s4 = self.att4(d4, e4)
		d4 = self.dec4(torch.cat([d4, s4], dim=1))

		d3 = self.up3(d4)
		d3 = self._match_spatial(d3, e3)
		s3 = self.att3(d3, e3)
		d3 = self.dec3(torch.cat([d3, s3], dim=1))

		d2 = self.up2(d3)
		d2 = self._match_spatial(d2, e2)
		s2 = self.att2(d2, e2)
		d2 = self.dec2(torch.cat([d2, s2], dim=1))

		d1 = self.up1(d2)
		d1 = self._match_spatial(d1, e1)
		s1 = self.att1(d1, e1)
		d1 = self.dec1(torch.cat([d1, s1], dim=1))

		head = self.output_head(d1)
		rgb_residual = torch.tanh(head[:, :3])
		confidence = torch.sigmoid(head[:, 3:4])

		frame_a = x[:, :3]
		frame_b = x[:, 3:6]
		forward_warped_a = self._forward_warp(frame_a)
		backward_warped_b = self._backward_warp(frame_b)

		blended = confidence * forward_warped_a + (1.0 - confidence) * backward_warped_b
		synthesized = torch.clamp(blended + 0.1 * rgb_residual, min=-1.0, max=1.0)
		return synthesized, confidence


if __name__ == "__main__":
	model = VFIUNet()
	test_input = torch.randn(1, 6, 256, 256)
	output_frame, output_conf = model(test_input)
	print(f"Synthesized frame shape: {output_frame.shape}")
	print(f"Confidence map shape: {output_conf.shape}")
	assert output_frame.shape == (1, 3, 256, 256)
