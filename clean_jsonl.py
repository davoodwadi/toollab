import json
import os
import tempfile
from pathlib import Path

def clean_jsonl(filepath):
    print(f"Cleaning {filepath}...")
    
    # Create a temporary file in the same directory to ensure atomic replacement
    fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(filepath), prefix="clean_", suffix=".jsonl")
    
    removed_count = 0
    kept_count = 0
    
    try:
        with open(filepath, 'r', encoding='utf-8') as infile, \
             os.fdopen(fd, 'w', encoding='utf-8') as outfile:
             
            for line in infile:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if data.get("budget_max") == 20:
                        removed_count += 1
                    else:
                        outfile.write(line)
                        kept_count += 1
                except json.JSONDecodeError:
                    # Keep any malformed lines so we don't accidentally delete other data
                    outfile.write(line)
                    kept_count += 1
        
        # Replace original file with the cleaned temporary file
        os.replace(temp_path, filepath)
        print(f"Done. Removed {removed_count} corrupted lines. Kept {kept_count} lines.")
        
    except Exception as e:
        os.remove(temp_path)
        print(f"Error processing {filepath}: {e}")

if __name__ == "__main__":
    target_dir = Path("/home/dw/github/toollab/results/promotion-discount-hidden")
    for file in target_dir.iterdir():
        if file.suffix=='.jsonl':
            print(file.name)
            clean_jsonl(file)