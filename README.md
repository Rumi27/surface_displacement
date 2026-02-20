# Automated Marker Detection Using YOLO for Soil Displacement Analysis

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of an automated marker detection system using YOLO (You Only Look Once) for analyzing soil displacement in large-scale shaking table tests. The system detects and tracks markers on soil surfaces to measure displacement during seismic simulations.

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{zafar2025automated,
  title={Automated Marker Detection Using YOLO for Soil Displacement Analysis in Large-Scale Shaking Table Tests},
  author={Zafar, A.},
  journal={Intelligence, Informatics and Infrastructure},
  volume={6},
  number={3},
  pages={137--149},
  year={2025}
}
```

## 🎯 Overview

This project implements a computer vision pipeline for:
- **Marker Detection**: Automated detection of markers on soil surfaces using YOLO (YOLOv5, YOLOv7, YOLOv8)
- **Displacement Calculation**: Measurement of marker displacement between before/after images
- **Data Augmentation**: Synthetic data generation using GANs and geometric transformations to prevent data leakage
- **Evaluation**: Comprehensive metrics including Pearson correlation, MAE, and SSIM

## 🏗️ Project Structure

```
surface_displacement/
├── calc_displacement.py          # Main displacement calculation script
├── compare_template_vs_yolo.py   # Template matching vs YOLO comparison
├── coordinates.py                 # Coordinate extraction utilities
├── check_duplicates.py            # Duplicate detection script
├── yolo/                          # YOLOv5 implementation and data
│   ├── calc_displacement.py      # YOLO-based displacement calculation
│   ├── evaluate_metrics.py       # Evaluation metrics
│   ├── heatmaps.py               # Displacement heatmap visualization
│   ├── fig6.py, fig7.py         # Figure generation scripts
│   ├── marker_yolo/              # YOLO dataset (synthetic/augmented images)
│   ├── aug_bank/                 # Augmented image bank
│   └── results_mapped/            # Mapped displacement results
├── yolov7/                        # YOLOv7 implementation
│   ├── yolov7/                   # YOLOv7 source code
│   ├── yolov7_data/              # Training data
│   └── aug_bank/                 # Augmented images
├── yolov8/                        # YOLOv8 implementation
├── prevent_data_leakage/         # Cross-validation setup
│   ├── case3/, case4/, case5/, case6/  # Per-case folders
│   │   ├── synth/images/         # GAN-generated synthetic images
│   │   ├── images/train/         # Training images
│   │   ├── images/val/           # Validation images
│   │   └── labels/               # YOLO format labels
│   └── scripts/                  # Data preparation scripts
├── tables/                        # Results tables
└── figures/                       # Generated figures
```

## 🔧 Installation

### Requirements

- Python 3.7 or higher
- PyTorch 1.7 or higher
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/surface_displacement.git
cd surface_displacement
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For YOLOv5, install from the subdirectory:
```bash
cd yolo/yolov5
pip install -r requirements.txt
cd ../..
```

4. For YOLOv7, install from the subdirectory:
```bash
cd yolov7
pip install -r requirements.txt
cd ..
```

## 📊 Usage

### 1. Training YOLO Models

#### YOLOv5
```bash
cd yolo/yolov5
python train.py --img 1920 --batch 16 --epochs 100 --data ../data.yaml --weights yolov5m.pt
```

#### YOLOv7
```bash
cd yolov7
python yolov7/train.py --img 1920 --batch 16 --epochs 100 --data data.yaml --weights yolov7.pt
```

### 2. Marker Detection

Run detection on images:
```bash
python yolo/yolov5/yolov5/detect.py --weights runs/train/exp/weights/best.pt --source path/to/images
```

### 3. Displacement Calculation

Calculate displacement between before/after image pairs:
```bash
python calc_displacement.py
```

This script:
- Loads trained YOLO model
- Detects markers in before/after image pairs
- Matches markers using nearest neighbor (KD-tree)
- Converts pixel displacements to millimeters
- Saves results to `results/` directory

### 4. Evaluation

Evaluate model performance:
```bash
python yolo/evaluate_metrics.py
```

Generate displacement heatmaps:
```bash
python yolo/heatmaps.py
```

### 5. Data Leakage Prevention

The `prevent_data_leakage/` directory contains scripts for cross-validation setup:

```bash
# Setup fold for case3 (holdout)
cd prevent_data_leakage
python scripts/setup_fold.py case3

# Generate synthetic images from training cases
python scripts/tmp_make_synth_from_training_cases.py --src case3/gan_train_input --out case3/synth/images --n 400

# Generate train/val split from synthetic images
python scripts/gen_split_from_synth.py case3
```

## 🔬 Methodology

### Marker Detection
- Uses YOLO object detection models (YOLOv5, YOLOv7, YOLOv8)
- Trained on synthetic and augmented images
- Single class detection: "marker"

### Displacement Measurement
1. Detect markers in before and after images
2. Match markers using nearest neighbor algorithm (KD-tree)
3. Calculate pixel displacement
4. Convert to real-world units (millimeters) using calibration

### Data Augmentation
- **Synthetic Images**: GAN-generated images from training cases
- **Geometric Augmentation**: Rotation, flipping, affine transformations
- **Photometric Augmentation**: Brightness, contrast, noise, JPEG compression

### Cross-Validation
- 4-fold cross-validation (Case3, Case4, Case5, Case6)
- Each case held out for testing
- Training on remaining cases + synthetic data

## 📈 Results

The system achieves displacement measurement with:
- **Pearson Correlation**: Varies by case (see `tables/displacement_metrics_summary.csv`)
- **Mean Absolute Error (MAE)**: 27-47 mm depending on case
- **SSIM**: Structural similarity metrics for displacement fields

## 📁 Data

**Note**: This repository does not include:
- Original test images (confidential)
- Actual ground truth measurements (confidential)

**Included**:
- Synthetic/GAN-generated images
- Augmented images
- Training/validation splits
- Processed results and figures

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- YOLOv5: [Ultralytics](https://github.com/ultralytics/yolov5)
- YOLOv7: [WongKinYiu](https://github.com/WongKinYiu/yolov7)
- YOLOv8: [Ultralytics](https://github.com/ultralytics/ultralytics)

## 📧 Contact

For questions or inquiries, please open an issue or contact the repository maintainer.

---

**Disclaimer**: The actual test images and measurements are not included in this repository due to confidentiality. The code and synthetic data are provided for reproducibility and research purposes.
