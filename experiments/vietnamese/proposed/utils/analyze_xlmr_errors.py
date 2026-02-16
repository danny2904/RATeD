import json
import pandas as pd
import numpy as np

def analyze_errors(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # False Positives (Clean labeled as Hate)
    fps = df[(df['true_label'] == 0) & (df['pred_label'] == 1)]
    # False Negatives (Hate labeled as Clean)
    fns = df[(df['true_label'] == 1) & (df['pred_label'] == 0)]
    
    # Low MF1 samples (Correct label but poor span extraction)
    low_mf1 = df[(df['true_label'] == 1) & (df['pred_label'] == 1) & (df['char_f1_macro'] < 0.5)]
    
    print(f"Total Samples: {len(df)}")
    print(f"False Positives: {len(fps)}")
    print(f"False Negatives: {len(fns)}")
    print(f"Low MF1 (Correct Class, Bad Span): {len(low_mf1)}")
    
    print("\n--- Top 5 False Positives ---")
    for i, row in fps.head(5).iterrows():
        print(f"ID: {row['id']} | Text: {row['text'][:100]}...")
        
    print("\n--- Top 5 False Negatives ---")
    for i, row in fns.head(5).iterrows():
        print(f"ID: {row['id']} | Text: {row['text'][:100]}...")
        
    print("\n--- Top 5 Low MF1 (Span Issues) ---")
    for i, row in low_mf1.head(5).iterrows():
        print(f"ID: {row['id']} | MF1: {row['char_f1_macro']:.4f}")
        print(f"Text: {row['text'][:100]}...")
        print(f"Gold: {row['gold_spans']}")
        print(f"Pred: {row['pred_spans']}")
        print("-" * 20)

if __name__ == "__main__":
    analyze_errors('experiments/vietnamese/results/test_predictions.json')
