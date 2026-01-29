# Bottle OCR & FDA Enforcement Pipeline

This repository contains an end-to-end pipeline for extracting medication information from bottle images, verifying it against the FDA database, and enriching it with safety data.

## 📂 Repository Structure
- `bottle_pipeline.py`: Main pipeline script (Detection -> OCR -> Validation -> Enrichment).
- `test_pipeline.ipynb`: Jupyter Notebook for batch testing and visual compliance reporting.
- `evaluate.py`: Script to calculate CER (Character Error Rate) for performance metrics.
- `best.pt`: Fine-tuned YOLOv8 model for medicine bottle detection.
- `data/`: Test images.

## 🏗 Architectural Summary

### 1. Detection & Extraction
-   **Object Detection**: Uses **YOLOv8** (`best.pt`) to localize the medicine bottle and crop the region of interest, ignoring background clutter.
-   **Resilient Preprocessing**:
    -   **CLAHE** (Contrast Limited Adaptive Histogram Equalization): Enhances local contrast to handle the non-linear lighting gradients typical of cylindrical bottles.
    -   **Upscaling**: Applies 2x Cubic Interpolation to improve OCR accuracy on fine print.
-   **Hybrid OCR Strategy**:
    -   **Layout-Aware Analysis**: Uses **Tesseract 5** (PSM 11 Sparse Text) to analyze text block geometry. It prioritizes the **largest/tallest** text blocks as the primary Drug Name candidates.
    -   **Robust Fallback**: Augments the layout analysis with a Regex layer (`[A-Z]{4,}`) to capture keywords that might be split or misaligned. This ensures high recall (e.g., catching "ANTOPRAZOLE").
-   **Barcode Decoding**: Uses **Pyzbar** with robust scanning (checking 0°, 90°, 180°, 270° rotations) to handle tilted bottles.

### 2. Verification (The Knowledge Link)
-   **Multi-Modal Validation**: The system treats Barcode and Text as independent truth sources.
    -   *Source A*: Barcode -> UPC Lookup -> FDA Brand Name.
    -   *Source B*: OCR -> NER/Heuristics -> FDA Brand Name.
    -   **Cross-Check Logic**: If both sources exist, they are compared. Discrepancies trigger a **"Mismatch Warning"**, ensuring data integrity.
-   **Fuzzy Entity Resolution**:
    -   **NER**: Uses `spacy` (`en_core_web_sm` fallback or `med7`).
    -   **Fuzzy Search**: FDA API queries use the `~` operator (e.g., `brand_name:"ANTOPRAZOLE"~2`) to resolve OCR typos and partial matches.

### 3. Enrichment
-   Once the drug is identified, the system queries the **openFDA API** to retrieve authoritative metadata:
    -   **Manufacturer Name** (e.g., "Meitheal Pharmaceuticals")
    -   **Active Ingredients** & **Strength**
    -   **Boxed Warnings**

## 📊 Performance & Evaluation
-   **Character Error Rate (CER)**: Use `evaluate.py` to calculate CER between OCR output and ground truth.
    ```bash
    python evaluate.py --reference "PANTOPRAZOLE" --hypothesis "ANTOPRAZOLE"
    ```
-   **Resilience**: The pipeline is designed to handle:
    -   **Curvature**: Via CLAHE and fuzzy matching.
    -   **Glare**: Via contrast limiting.
    -   **Layout Variation**: Via the font-size heuristic (no hardcoded coordinates).

## 🚀 Usage

### Installation
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Run Pipeline
**Single Image:**
```bash
python bottle_pipeline.py --image "data/test/images/sample.jpg"
```

**Batch Directory:**
```bash
python bottle_pipeline.py --dir "data/test/images/"
```

**Jupyter Report:**
Open `test_pipeline.ipynb` to visualize the "Compliance Report" for all images.
