import json
import os

def verify_splits():
    divisions_path = r'c:\Projects\RATeD-V\temp_hatexplain\Data\post_id_divisions.json'
    prepared_path = r'c:\Projects\RATeD-V\experiments\english\data\hatexplain_prepared.jsonl'

    if not os.path.exists(divisions_path):
        print(f"Error: Divisions file not found at {divisions_path}")
        return
    if not os.path.exists(prepared_path):
        print(f"Error: Prepared file not found at {prepared_path}")
        return

    with open(divisions_path, 'r', encoding='utf-8') as f:
        divisions = json.load(f)

    # Create a mapping from ID to expected split
    id_to_expected_split = {}
    for split_name, ids in divisions.items():
        for post_id in ids:
            id_to_expected_split[post_id] = split_name

    matches = 0
    mismatches = 0
    missing_in_divisions = 0
    found_ids = set()

    with open(prepared_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            post_id = item.get('id')
            actual_split = item.get('split')
            found_ids.add(post_id)

            if post_id in id_to_expected_split:
                expected_split = id_to_expected_split[post_id]
                if actual_split == expected_split:
                    matches += 1
                else:
                    mismatches += 1
                    # print(f"Mismatch for {post_id}: expected {expected_split}, got {actual_split}")
            else:
                missing_in_divisions += 1
                # print(f"ID {post_id} from prepared file not found in divisions.json")

    all_expected_ids = set(id_to_expected_split.keys())
    missing_in_prepared = all_expected_ids - found_ids

    print("--- Split Verification Results ---")
    print(f"Total items in prepared file: {len(found_ids)}")
    print(f"Matches with divisions.json: {matches}")
    print(f"Mismatches: {mismatches}")
    print(f"IDs in prepared but NOT in divisions.json: {missing_in_divisions}")
    print(f"IDs in divisions.json but NOT in prepared: {len(missing_in_prepared)}")
    
    if mismatches == 0 and len(missing_in_prepared) == 0:
        print("\nSUCCESS: Splits are aligned perfectly.")
    else:
        print("\nFAILURE: Splits are NOT aligned.")

if __name__ == "__main__":
    verify_splits()
