# VFI - Video Frame Interpolation

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows-0078D6?logo=windows&logoColor=white)

VFI is a video frame interpolation project with:

- A desktop app UI for running interpolation on videos.
- A training pipeline for learning a middle-frame predictor from triplet frames `(A, B, C)`.
- Dataset preparation and evaluation scripts.
- PyInstaller packaging support for a Windows executable.

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Prepare Training Data](#prepare-training-data)
- [Train the Model](#train-the-model)
- [Evaluate a Trained Model](#evaluate-a-trained-model)
- [Run the Desktop App](#run-the-desktop-app)
- [Build Windows Executable](#build-windows-executable)
- [Troubleshooting](#troubleshooting)

## Project Overview

The model is trained on triplets of frames where:

- `A.png` is the first frame.
- `B.png` is the ground-truth middle frame.
- `C.png` is the third frame.

Training uses a combined loss with pixel reconstruction, edge consistency, and confidence-map supervision. Checkpoints are saved every epoch, and the best validation model is written to `models/best.pth`.

## Repository Layout

```text
app.py
data/
	raw_videos/
	triplets/
models/
notebooks/
	train_colab.ipynb
scripts/
	download_data.py
	evaluate.py
	prepare_dataset.py
	validate_videos.py
	build.py
src/
	dataset.py
	model.py
	predict.py
	train.py
	video_utils.py
tests/
vfi.spec
```

## Prerequisites

- Python 3.10 or newer
- pip
- Windows (for `.exe` packaging)
- Optional GPU (CUDA) for faster training

## Installation

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

Install training dependencies:

```bash
pip install -r requirements-train.txt
```

Optional (for packaging):

```bash
pip install pyinstaller
```

## Prepare Training Data

### Option A: Download sample raw videos from Pexels

Set your API key, then download videos:

```bash
set PEXELS_API_KEY=your_api_key_here
python scripts/download_data.py --count 30
```

### Option B: Add your own videos

Place videos (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`) in:

```text
data/raw_videos/
```

### Validate videos

```bash
python scripts/validate_videos.py
```

Invalid videos are moved into `data/raw_videos/rejected/`.

### Build frame triplets dataset

```bash
python scripts/prepare_dataset.py
```

This generates subfolders in `data/triplets/`, each containing `A.png`, `B.png`, and `C.png`.

## Train the Model

The training entrypoint is the `train(config)` function in `src/train.py`.

### Quick local training command

Run this one-liner from the project root:

```bash
python -c "from src.train import train; train({'data_dir':'data/triplets','epochs':50,'batch_size':8,'learning_rate':1e-4,'checkpoint_dir':'models','device':'cuda'})"
```

If you do not have a GPU, use `device='cpu'`.

### Recommended config fields

- `data_dir`: triplet dataset path (usually `data/triplets`)
- `epochs`: number of training epochs
- `batch_size`: training batch size
- `learning_rate`: optimizer learning rate
- `checkpoint_dir`: checkpoint output directory (usually `models`)
- `device`: `cuda` or `cpu`

### Training outputs

- Per-epoch checkpoints: `models/epoch_<N>.pth`
- Best checkpoint: `models/best.pth`

## Evaluate a Trained Model

Run evaluation on a test triplet directory:

```bash
python scripts/evaluate.py --model_path models/best.pth --test_dir data/triplets --output_dir evaluation_output --device cpu
```

This prints PSNR/SSIM metrics and writes visual comparisons to `evaluation_output/`.

## Run the Desktop App

Start the app:

```bash
python app.py
```

In the app, select:

- Input video
- Output path
- Model weights (default points to `models/best.pth`)

## Build Windows Executable

### Prerequisites

```bash
pip install pyinstaller
```

### Build

```bash
python scripts/build.py
```

### Output location

Final executable:

```text
dist/VFI.exe
```

### Important model note

Ensure `models/best.pth` exists before building (the spec bundles this file into the executable).

## Colab Training (Optional)

You can also train using the notebook in `notebooks/train_colab.ipynb`.

High-level flow:

1. Install dependencies in Colab.
2. Mount Google Drive.
3. Clone the repository.
4. Copy triplet data into `data/triplets`.
5. Run training via `from src.train import train`.
6. Copy `models/best.pth` back to Drive.

## Troubleshooting

- `No valid triplet subfolders found`: run `python scripts/prepare_dataset.py` and confirm triplets exist.
- `Model file not found`: verify `models/best.pth` exists.
- Slow training: reduce `batch_size` on limited hardware or use GPU.
- Packaging issues: reinstall PyInstaller and rebuild with `python scripts/build.py`.
