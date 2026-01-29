# CartEye: Supermarket Checkout Automation 🛒👁️

Automating supermarket checkout using computer vision for the **Vista’26 Hackathon (IIT BHU)**.

## Project Overview
CartEye is an end-to-end computer vision pipeline designed to accurately detect and count supermarket products on a checkout counter. It addresses real-world challenges like **overlapping objects**, **crowded scenes**, and the need for **real-time inference** on edge devices.

## Key Features
- **High-Accuracy Counting**: Optimized thresholds (Grid Search) for exact product count matching.
- **Handling Overlaps**: Utilizes **YOLOv10** (NMS-free) and **Test-Time Augmentation (TTA)** to catch hidden items.
- **Mac Optimized**: Achieving **85+ FPS** on M3 hardware via **CoreML FP16 Quantization**.
- **Robust Submission Pipeline**: Automated CSV generation with exact integer ID mapping and sorting.

## Documentation
- [Technical Pipeline Report](file:///Users/khushchaudhari/Documents/IITB_HACKATHAON/CartEye/docs/TECHNICAL_REPORT.md): Deep dive into data engineering, model selection, and optimizations.
- [Submission Guide](file:///Users/khushchaudhari/Documents/IITB_HACKATHAON/CartEye/docs/SUBMISSION_GUIDE.md): How to generate your official Kaggle submission.

## Project Structure
- `data/`: Automated YOLO conversion and subset management.
- `docs/`: Comprehensive technical documentation.
- `models/`: Weights (.pt) and hardware-optimized binaries (.mlpackage).
- `scripts/`: Production-ready training, inference, and export tools.

## Quick Start
1. **Setup**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Generate Submission**:
   Update paths in `scripts/final_submission.py` and run.
