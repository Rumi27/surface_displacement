# Quick Start Guide

This guide will help you get started with the Automated Marker Detection system quickly.

## Prerequisites

- Python 3.7 or higher
- CUDA-capable GPU (recommended, but CPU will work)
- 8GB+ RAM
- 10GB+ free disk space

## Installation (5 minutes)

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/surface_displacement.git
cd surface_displacement
```

2. **Create and activate virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install YOLOv5 (if using YOLOv5):**
```bash
cd yolo/yolov5
pip install -r requirements.txt
cd ../..
```

## Basic Usage

### Step 1: Prepare Your Data

1. Organize your images in the following structure:
```
your_data/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

2. Update `data.yaml` with your paths (or use the template):
```yaml
train: your_data/images/train
val: your_data/images/val
nc: 1
names: ['marker']
```

### Step 2: Train a Model

**Using YOLOv5:**
```bash
cd yolo/yolov5
python train.py --img 1920 --batch 16 --epochs 100 --data ../../data.yaml --weights yolov5m.pt
```

**Using YOLOv7:**
```bash
cd yolov7
python yolov7/train.py --img 1920 --batch 16 --epochs 100 --data data.yaml --weights yolov7.pt
```

### Step 3: Run Detection

```bash
# Using YOLOv5
cd yolo/yolov5
python detect.py --weights runs/train/exp/weights/best.pt --source path/to/images --conf 0.25
```

### Step 4: Calculate Displacement

1. Place your before/after image pairs in a directory (e.g., `test_images/`)
2. Update `calc_displacement.py` with your model path and image directory
3. Run:
```bash
python calc_displacement.py
```

Results will be saved in the `results/` directory.

## Example Workflow

```bash
# 1. Setup
git clone https://github.com/yourusername/surface_displacement.git
cd surface_displacement
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Train (example with YOLOv5)
cd yolo/yolov5
python train.py --img 1920 --batch 16 --epochs 50 --data ../../data.yaml --weights yolov5m.pt

# 3. Detect markers
python detect.py --weights runs/train/exp/weights/best.pt --source ../../test_images

# 4. Calculate displacement
cd ../..
python calc_displacement.py
```

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'ultralytics'`
- **Solution**: Install ultralytics: `pip install ultralytics`

**Issue**: CUDA out of memory
- **Solution**: Reduce batch size: `--batch 8` or `--batch 4`

**Issue**: Path errors
- **Solution**: Use relative paths in `data.yaml` or update absolute paths

**Issue**: No detections
- **Solution**: Lower confidence threshold: `--conf 0.15`

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [CONTRIBUTING.md](CONTRIBUTING.md) if you want to contribute
- Review the code in `calc_displacement.py` to understand the displacement calculation

## Getting Help

- Open an issue on GitHub for bugs or questions
- Check existing issues for similar problems
- Review the paper for methodology details

Happy detecting! 🎯
