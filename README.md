# VFI — Video Frame Interpolation

VFI — Video Frame Interpolation is a Python project for building and running a video frame interpolation workflow, with a lightweight app entry point, model code, dataset handling, and training and prediction scripts organized for future development.

## Setup

Install the runtime dependencies with:

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

## Training

Training notes will be added here later.

## Build

### Prerequisites

```bash
pip install pyinstaller
```

### Build the executable

```bash
python scripts/build.py
```

### Output

The final executable is created at `dist/VFI.exe`.

### Model file requirement

`vfi.pth` must be in `models/` before building.
