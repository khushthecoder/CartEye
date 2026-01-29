# CartEye: Final Submission Guide

Follow these steps to generate your official Kaggle submission CSV file for the Vista'26 Hackathon.

## Prerequisites
- A trained model (e.g., `runs/detect/train/weights/best.pt`).
- Access to the raw competition dataset (test images and JSON metadata).

## Generation Steps

### 1. Configure the Script
Open [scripts/final_submission.py](file:///Users/khushchaudhari/Documents/IITB_HACKATHAON/CartEye/scripts/final_submission.py) and update the following variables:
```python
# Path to your final trained model
MODEL = "models/best_production.pt" 

# Path to the competition TEST images directory
IMAGES = "/path/to/vistas/test/images"

# Path to the competition TEST JSON metadata
JSON_MAP = "/path/to/vistas/instances_test.json"
```

### 2. Run the Generator
Execute the script from the project root:
```bash
python3 scripts/final_submission.py
```

### 3. Verify Output
The script will produce `final_submission.csv` in the project root.
- **Total Rows**: Should match the number of images in the `test` set.
- **Formatting**:
    - `image_id`: Integer.
    - `categories`: JSON-style list (e.g., `[22, 114]`).
    - **Sorting**: The IDs within the list and the rows themselves are automatically sorted.

## Using the Optimized Model (Optional)
For ultra-fast generation on a Mac, you can use the CoreML model exported in Issue 11:
1. Update `MODEL` in `final_submission.py` to point to `best.mlpackage`.
2. Ensure you have `coremltools` installed (`pip install coremltools`).

## Accuracy Tips
Ensure you use the same **imgsz** for inference that was used in training unless using the high-res strategy defined in `scripts/inference_optimized.py`.
