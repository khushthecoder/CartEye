# CartEye: Technical Pipeline & Design Report

## 1. Project Objective
Automate supermarket checkout by detecting items on a counter using computer vision. Our solution prioritizes **exact counting accuracy**, **handling of overlapping objects**, and **production-ready speed**.

## 2. Data Engineering
### Processing & Mapping
- **COCO to YOLO Conversion**: Automated the translation of raw COCO annotations into YOLOv8 format for high-speed training.
- **Categorization**: Handled 114 product categories with exact integer mapping ensured via `metadata['categories']` extraction.

### Data Augmentations
To handle varied supermarket lighting and shelf arrangements, we applied:
- **Mosaic & Mixup**: Improves detection of small and partially occluded objects.
- **HSV & Blur**: Increases model robustness against sensor noise and lighting shifts.
- **Scale & Degrees**: Ensures the model is invariant to product orientation on the counter.

### Iteration Strategy
- Created a **10% Representative Subset** for rapid training cycles (1 epoch tests), allowing us to iterate on 12 distinct issues within a limited timeframe.

## 3. Model Architecture & Selection
### Baseline: YOLOv8n (Nano)
- **Rationale**: Chosen for its extreme lightness (3.2M params) and speed on edge hardware. Established the project's mAP baseline.

### Specialized Models: YOLOv10 & RT-DETR
- **YOLOv10n**: Introduced an **NMS-free** design. This is critical for supermarket counters where products are frequently stacked or overlapping. By removing the Non-Maximum Suppression bottleneck during training, we reduce "over-suppression" of real objects.
- **RT-DETR-L**: Explored for its transformer-based global context, which helps in identifying products by their proximity to others.

## 4. Advanced Optimizations
### Hardware Acceleration (Mac M3)
- Utilized **MPS (Metal Performance Shaders)** for training and inference, achieving near-GPU speeds on native Apple Silicon.

### Accuracy Boosters (Issue 9)
- **TTA (Test-Time Augmentation)**: The model performs inference on multiple augmented versions of an image, merging results to miss fewer occluded items.
- **High-Res Inference**: Inference is performed at **640px** (2x training resolution) to provide higher pixel density for distinguishing between small, similar-looking packages.

### Threshold Tuning (Issue 6)
- Performed a **Grid Search** over Confidence (0.1 - 0.5) and IoU (0.4 - 0.8) thresholds.
- **Goal**: Minimize **Mean Absolute Error (MAE)** in counts rather than just maximizing mAP, directly aligning with the competition's "Exact Count" scoring.

## 5. Production Readying (Issue 11)
### Model Quantization
Converted our weights to **FP16 (Half-Precision)**, resulting in:
- **Size**: 19.7 MB → **6.6 MB** (~66% reduction).
- **Format**: Exported to **CoreML (.mlpackage)** for direct execution on the Mac Neural Engine.

### Benchmarks (Mac M3)
| Model | Latency | FPS |
| :--- | :--- | :--- |
| PyTorch (.pt) | 92.2 ms | 10.8 |
| **CoreML (Quantized)** | **11.7 ms** | **85.6** |

## 6. Submission Pipeline
### Robust CSV Generation
- **Script**: `scripts/final_submission.py`.
- **Integrity**: Ensures all 6000 validation/test images are accounted for, with sorted category IDs and sorted image IDs, guaranteeing a "Successful Submission" on the leaderboard.
