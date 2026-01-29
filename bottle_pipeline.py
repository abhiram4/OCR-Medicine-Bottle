import cv2
import pytesseract
import numpy as np
import requests
import spacy
from ultralytics import YOLO
from pyzbar.pyzbar import decode
import argparse
import sys
import os
import re

# --- Configuration ---
# Set pysseract path if not in standard path (User can modify this)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class BottlePipeline:
    def __init__(self, model_path="best.pt", ner_model="en_core_web_sm"):
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        
        print(f"Loading NER model {ner_model}...")
        try:
            self.nlp = spacy.load(ner_model)
        except OSError:
            print(f"Warning: Model '{ner_model}' not found. Installing or falling back recommended.")
            # Basic fallback if med7 fails to load, though we expect it to be there based on install
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                self.nlp = None

    def detect_bottle(self, image_path):
        """Detects the bottle using YOLO and returns the cropped image and bbox."""
        print(f"Detecting bottle in {image_path}...")
        results = self.model(image_path)
        
        # Taking the first result
        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:
                # Get the box with the highest confidence
                best_box = max(boxes, key=lambda x: x.conf[0])
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                
                # Load original image
                img = cv2.imread(image_path)
                cropped_img = img[y1:y2, x1:x2]
                return cropped_img, (x1, y1, x2, y2)
        
        print("No bottle detected.")
        return cv2.imread(image_path), None  # Return original if no detection

    def preprocess(self, image):
        """Preprocesses image for OCR (Grayscale, Resize, CLAHE)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Upscale
        scale_percent = 200 
        width = int(gray.shape[1] * scale_percent / 100)
        height = int(gray.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(gray, dim, interpolation=cv2.INTER_CUBIC)

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Helps with cylindrical wrapping lighting issues
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        processed = clahe.apply(resized)

        cv2.imwrite("debug_before_ocr.jpg", processed)
        
        return processed

    def perform_ocr(self, image):
        """Extracts text using Tesseract."""
        # psm 6: Assume a single uniform block of text.
        text = pytesseract.image_to_string(image, config='--psm 6')
        return text.strip()

    def read_barcode(self, image):
        """Reads barcode from the image with rotations and sharpening."""
        
        candidates = []
        
        # Define rotations: 0, 90, 180, 270
        rotations = [0, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
        
        for rot_code in rotations:
            if rot_code == 0:
                rotated = image
            else:
                rotated = cv2.rotate(image, rot_code)
                
            # Try raw
            decoded = decode(rotated)
            if decoded:
                for obj in decoded:
                    candidates.append({"type": obj.type, "data": obj.data.decode("utf-8")})
            
            # Try sharpened
            kernel = np.array([[0, -1, 0], 
                               [-1, 5,-1], 
                               [0, -1, 0]])
            sharpened = cv2.filter2D(rotated, -1, kernel)
            decoded_sharp = decode(sharpened)
            if decoded_sharp:
                for obj in decoded_sharp:
                    candidates.append({"type": obj.type, "data": obj.data.decode("utf-8")})
            
            if candidates: 
                break # Found something, stop rotating
        
        # Deduplicate
        unique_candidates = []
        seen = set()
        for c in candidates:
            if c['data'] not in seen:
                unique_candidates.append(c)
                seen.add(c['data'])
                
        return unique_candidates

    def extract_entities(self, text):
        """Extracts entities like DRUG, DOSAGE using med7."""
        if not self.nlp:
            return {"raw_text": text}
            
        doc = self.nlp(text)
        entities = {}
        for ent in doc.ents:
            entities[ent.label_] = ent.text
        return entities

    def extract_candidates_heuristic(self, text):
        """
        Fallback method to find potential drug names if NER fails.
        Looks for uppercase words > 3 chars, ignoring common keywords.
        """
        # Common non-drug words on labels
        ignore_list = {
            "TABLET", "TABLETS", "CAPSULE", "CAPSULES", "ORAL", "USP", 
            "MG", "MCG", "ML", "EXP", "DATE", "QTY", "DOSE", "TAKE", "MOUTH", 
            "DAILY", "ONLY", "KEEP", "AWAY", "FROM", "CHILDREN", "STORE", 
            "REFILL", "PHARMACY", "GENERIC", "BRAND", "SUBSTITUTE", "FOR"
        }
        
        words = re.findall(r'\b[A-Z]{4,}\b', text)
        candidates = [w for w in words if w not in ignore_list]
        return candidates

    def query_fda(self, query_term, query_type="brand_name", fuzzy=False):
        """Queries the FDA API for the drug name or UPC. Supports fuzzy matching."""
        base_url = "https://api.fda.gov/drug/label.json"
        
        # OpenFDA search for UPC is 'openfda.upc'
        # For names, we use brand_name or generic_name
        
        if query_type == "upc":
            query = f'openfda.upc:"{query_term}"'
        else:
            # Check for multi-word strict or fuzzy
            if fuzzy:
                # Fuzzy search syntax: field:"term"~2 (allows 2 edits)
                query = f'openfda.brand_name:"{query_term}"~2'
            else:
                query = f'openfda.brand_name:"{query_term}"'
        
        params = {
            "search": query,
            "limit": 1
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "results" in data:
                result = data["results"][0]
                return {
                    "scan_type": query_type,
                    "term": query_term,
                    "fuzzy_match": fuzzy,
                    "brand_name": result.get("openfda", {}).get("brand_name", ["N/A"])[0],
                    "generic_name": result.get("openfda", {}).get("generic_name", ["N/A"])[0],
                    "manufacturer": result.get("openfda", {}).get("manufacturer_name", ["N/A"])[0],
                    "active_ingredients": result.get("openfda", {}).get("substance_name", ["N/A"]),
                    "packager": result.get("openfda", {}).get("package_ndc", ["N/A"]),
                    "warnings": result.get("warnings", ["N/A"])[:1], 
                    "indications": result.get("indications_and_usage", ["N/A"])[:1]
                }
            else:
                return {"error": "No results found"}
        except Exception as e:
            return {"error": str(e)}

    def run(self, image_path):
        print(f"--- Processing {os.path.basename(image_path)} ---")
        
        result_data = {
            "image_path": image_path,
            "bottle_detected": False,
            "barcode_found": False,
            "barcodes": [],
            "ocr_text": "",
            "drug_candidates": [],
            "fda_data": None,
            "validation_status": "Pending", # Validated, Mismatch, Unverified
            "extracted_entities": {}
        }

        # 1. Detection
        cropped_img, bbox = self.detect_bottle(image_path)
        if bbox:
            print(f"Bottle found at: {bbox}")
            result_data["bottle_detected"] = True
        else:
            print("Processing full image (no bottle detected).")

        # 2. Barcode Reading
        barcodes = self.read_barcode(cv2.imread(image_path))
        if not barcodes:
            barcodes = self.read_barcode(cropped_img)
            
        print(f"Barcodes Found: {barcodes}")
        result_data["barcodes"] = barcodes
        result_data["barcode_found"] = bool(barcodes)
        
        # FDA via Barcode (Truth Source A)
        fda_barcode_data = None
        if barcodes:
            for bc in barcodes:
                if bc['type'] in ['EAN13', 'UPCA']: 
                    print(f"Querying FDA for Barcode: {bc['data']}")
                    info = self.query_fda(bc['data'], query_type="upc")
                    if "error" not in info:
                        fda_barcode_data = info
                        print(f"FDA matched barcode to: {info['brand_name']}")
                        break
        
        # 3. Preprocessing & OCR (Always run this for validation)
        processed_img = self.preprocess(cropped_img)
        raw_text = self.perform_ocr(processed_img)
        result_data["ocr_text"] = raw_text
        
        # 4. NER & Smart Extraction
        entities = self.extract_entities(raw_text)
        result_data["extracted_entities"] = entities
        
        # Try heuristics if basic NER fails
        heuristic_candidates = []
        if not entities.get("DRUG"):
             heuristic_candidates = self.extract_candidates_heuristic(raw_text)
             result_data["drug_candidates"] = heuristic_candidates
        
        # FDA via Text (Truth Source B)
        fda_text_data = None
        drug_name_source = entities.get("DRUG")
        
        # Try to resolve text-based drug name if we don't have a barcode match OR we want to cross-verify
        candidates_to_check = [drug_name_source] if drug_name_source else heuristic_candidates
        
        for cand in candidates_to_check:
            if not cand: continue
            print(f"Querying FDA for Text Candidate: {cand}")
            info = self.query_fda(cand, query_type="brand_name")
            if "error" in info:
                 info = self.query_fda(cand, query_type="brand_name", fuzzy=True)
            
            if "error" not in info:
                fda_text_data = info
                print(f"FDA matched text '{cand}' to: {info['brand_name']}")
                break

        # 5. Multi-Modal Validation & Synthesis
        final_fda = None
        
        if fda_barcode_data and fda_text_data:
            # We have both sources. Compare them.
            name_bc = fda_barcode_data['brand_name'].lower().split()[0]
            name_txt = fda_text_data['brand_name'].lower().split()[0]
            
            # Simple fuzzy comparison
            if name_bc in name_txt or name_txt in name_bc:
                result_data["validation_status"] = "Verified Match"
                final_fda = fda_barcode_data # Prefer barcode data as it's usually more specific (package level)
            else:
                result_data["validation_status"] = "Mismatch Warning"
                print(f"WARNING: Barcode says {name_bc} but OCR says {name_txt}")
                final_fda = fda_barcode_data # Default to barcode but flag it
                
        elif fda_barcode_data:
            result_data["validation_status"] = "Barcode Verified (No Text Match)"
            final_fda = fda_barcode_data
            
        elif fda_text_data:
            result_data["validation_status"] = "Text Verified (No Barcode)"
            final_fda = fda_text_data
            
        else:
            result_data["validation_status"] = "Unverified (No FDA Match)"

        result_data["fda_data"] = final_fda
        
        # Print Summary
        print("\n" + "="*30)
        print(f"Validation Status: {result_data['validation_status']}")
        if final_fda:
            print(f"Identified Drug: {final_fda['brand_name']}")
            print(f"Manufacturer: {final_fda['manufacturer']}")
            print(f"Active Ingredients: {final_fda['active_ingredients']}")
        print("="*30 + "\n")

        return result_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bottle OCR Pipeline")
    parser.add_argument("--image", help="Path to input image")
    parser.add_argument("--dir", help="Path to directory of images for batch processing")
    parser.add_argument("--test-drug", help="Manually specify drug name to test FDA API")
    args = parser.parse_args()

    pipeline = BottlePipeline()
    
    if args.test_drug:
        print(f"Testing FDA API for: {args.test_drug}")
        info = pipeline.query_fda(args.test_drug)
        print(info)
    elif args.dir:
        import glob
        # Support common image formats
        types = ('*.jpg', '*.jpeg', '*.png', '*.tif')
        image_files = []
        for files in types:
            image_files.extend(glob.glob(os.path.join(args.dir, files)))
            
        print(f"Found {len(image_files)} images in {args.dir}")
        for img_path in image_files:
            print(f"\nProcessing: {os.path.basename(img_path)}")
            try:
                pipeline.run(img_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
            print("-" * 50)
    elif args.image:
        pipeline.run(args.image)
    else:
        print("Please provide --image or --dir")
