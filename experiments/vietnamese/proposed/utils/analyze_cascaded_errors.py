import json
import os

results_path = "experiments/vietnamese/proposed/results/cascaded_results.json" # Updated to default output
with open(results_path, "r", encoding="utf-8") as f:
    data = json.load(f)

fn_samples = []
recovered_fn = []
missed_fn = []

for item in data:
    true_label = 1 if item["true_label"] == "unsafe" else 0
    pred_label = 1 if item["pred_label"] == "hate" else 0
    
    if true_label == 1 and pred_label == 0:
        missed_fn.append(item)
    elif true_label == 1 and pred_label == 1 and item["flow"] == "JUDGE_RECOVERED_FN":
        recovered_fn.append(item)

print(f"Total True Hate Samples: {len([x for x in data if x['true_label'] == 'unsafe'])}")
print(f"Total Recovered FN: {len(recovered_fn)}")
print(f"Total Missed FN: {len(missed_fn)}")

print("\n--- EXAMPLES OF MISSED FN ---")
for item in missed_fn[:5]:
    print(f"Text: {item['text']}")
    print(f"Gold Spans: {item['gold_spans']}")
    print(f"Confidence: {item['rated_confidence']:.4f}")
    print(f"Flow: {item['flow']}")
    print("-" * 20)

print("\n--- EXAMPLES OF RECOVERED FN ---")
for item in recovered_fn:
    print(f"Text: {item['text']}")
    print(f"Pred Spans: {item['pred_spans']}")
    print(f"Confidence: {item['rated_confidence']:.4f}")
    print("-" * 20)
