
import json
import os

def load_bias_metadata_from_prepared(prepared_path, results_len):
    """
    Load annotator metadata from prepared jsonl file for bias calculation.
    Returns list of items with 'annotators' and 'label' matching the results order.
    Assumes results correspond to the 'test' split in the prepared file.
    """
    if not os.path.exists(prepared_path):
        print(f"⚠️ Prepared file {prepared_path} not found.")
        return []

    bias_items = []
    with open(prepared_path, 'r', encoding='utf-8') as f:
        # Filter for test split
        for line in f:
            try:
                item = json.loads(line)
                if item.get('split') == 'test':
                    # Extract only necessary fields
                    bias_items.append({
                        'annotators': item.get('annotators', []),
                        'label': item.get('label')
                    })
            except: pass
            
    # Truncate or check length
    if len(bias_items) < results_len:
        print(f"⚠️ Warning: Found {len(bias_items)} test items with metadata, but have {results_len} results. Bias metrics may be misaligned.")
    
    return bias_items[:results_len]
