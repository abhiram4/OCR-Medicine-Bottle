# Bottle OCR & FDA Enforcement Pipeline

This repository contains an end-to-end pipeline for extracting medication information from bottle images, verifying it against the FDA database, and enriching it with safety data.

## 📂 Repository Structure
- `bottle_pipeline.py`: Main pipeline script (Detection -> OCR -> Validation -> Enrichment).
- `test_pipeline.ipynb`: Jupyter Notebook for batch testing and visual verification.
- `evaluate.py`: Script to calculate CER (Character Error Rate) and WER (Word Error Rate).
- `best.pt`: Fine-tuned YOLOv8 model for bottle detection.
- `data/`: Test images.

## 🏗 Architectural Summary

### 1. Detection & Extraction
-   **Object Detection**: Uses **YOLOv8** (`best.pt`) to localize the medicine bottle and crop the region of interest. This handles complex backgrounds.
-   **Correction & Preprocessing**:
    -   **CLAHE** (Contrast Limited Adaptive Histogram Equalization) is applied to mitigate specular glare and lighting gradients relative to cylindrical bottles.
    -   **Upscaling** (2x) is performed to improve OCR on small fonts.
-   **OCR Engine**: **Tesseract 5** (via `pytesseract`) with Page Segmentation Mode 6 (Sparse Text).
-   **Barcode Decoding**: **Pyzbar** with robust scanning (0°, 90°, 180°, 270° rotations + sharpening) to handle orientation and blur.

### 2. Verification (The Knowledge Link)
-   **Multi-Modal Validation**: The system treats Barcode and Text as independent truth sources.
    -   *Source A*: Barcode -> UPC Lookup -> FDA Brand Name.
    -   *Source B*: OCR -> NER/Heuristics -> FDA Brand Name.
    -   **Logic**: If both exist, they are cross-referenced. Discrepancies trigger a "Mismatch Warning".
-   **Fuzzy Entity Resolution**:
    -   **NER**: Uses `spacy` (Medical `med7` or `en_core_web_sm` fallback).
    -   **Heuristics**: If NER fails, regex extracts "drug-like" candidates (uppercase, length > 4).
    -   **Fuzzy Search**: FDA API queries use the `~` operator to match typo-ridden OCR (e.g., "ANTOPRAZOLE" -> "PANTOPRAZOLE").

### 3. Enrichment
-   Once the drug is identified, the system queries the **openFDA API** to retrieve:
    -   **Manufacturer Name**
    -   **Active Ingredients**
    -   **Boxed Warnings** & **Indications**

## 📊 Performance Report

### Metric Definitions
-   **CER (Character Error Rate)**: `(S + D + I) / N`
    - Measures how many characters (substitutions, deletions, insertions) are needed to transform the OCR output to the Ground Truth.
-   **Entity Match Rate**: Percentage of images where the correct Drug Name was identified (either via Barcode or OCR).

### Resilience
-   **Curvature**: Handled via CLAHE and fuzzy matching (tolerates character loss on edges).
-   **Glare**: Addressed by adaptive thresholding/CLAHE.
-   **Layout Agnosticism**: The pipeline does not filter by coordinates. It uses entity recognition (NER) to find the drug name anywhere on the label.

## 🚀 Usage

### Installation
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
# Optional: Install med7 for better medical NER
# pip install https://med7.s3.eu-west-2.amazonaws.com/en_core_med7_lg.tar.gz
```

### Run Pipeline
```bash
# Process a single image
python bottle_pipeline.py --image data/test/images/sample.jpg

# Process a directory
python bottle_pipeline.py --dir data/test/images/
```

### Evaluation
```bash
# Calculate CER (requires ground truth JSON)
python evaluate.py --ground-truth matches.json
```
