import argparse
import json
import sys

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def calculate_cer(reference, hypothesis):
    """
    Calculate Character Error Rate (CER).
    CER = (S + D + I) / N
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0
        
    dist = levenshtein(reference, hypothesis)
    return dist / len(reference)

def main():
    parser = argparse.ArgumentParser(description="Calculate CER/WER for OCR results.")
    parser.add_argument("--reference", help="Ground truth text string")
    parser.add_argument("--hypothesis", help="OCR output text string")
    parser.add_argument("--file", help="JSON file with List of {reference: str, hypothesis: str}")
    
    args = parser.parse_args()

    if args.file:
        try:
            with open(args.file, 'r') as f:
                data = json.load(f)
            
            total_cer = 0
            count = 0
            
            print(f"{'Reference':<30} | {'Hypothesis':<30} | {'CER':<10}")
            print("-" * 80)
            
            for item in data:
                ref = item.get('reference', '')
                hyp = item.get('hypothesis', '')
                cer = calculate_cer(ref, hyp)
                total_cer += cer
                count += 1
                print(f"{ref[:28]:<30} | {hyp[:28]:<30} | {cer:.2f}")
            
            if count > 0:
                print("-" * 80)
                print(f"Average CER: {total_cer / count:.4f}")
                
        except Exception as e:
            print(f"Error reading file: {e}")
            
    elif args.reference and args.hypothesis:
        cer = calculate_cer(args.reference, args.hypothesis)
        print(f"Reference:  {args.reference}")
        print(f"Hypothesis: {args.hypothesis}")
        print(f"CER:        {cer:.4f}")
    else:
        print("Please provide --reference and --hypothesis OR --file")

if __name__ == "__main__":
    main()
