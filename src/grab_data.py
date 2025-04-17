from datasets import load_dataset
import json
import os
import random

print("Loading datasets...")

# Load split_0 dataset only (since split_1 is causing errors)
ocr_ds_split_0 = load_dataset("nvidia/OpenCodeReasoning", "split_0")
print(ocr_ds_split_0)

# Limit to 500 entries
print("Processing 500 entries from split_0 dataset...")
# Select 500 random entries
all_indices = list(range(len(ocr_ds_split_0["split_0"])))
random.shuffle(all_indices)
selected_indices = all_indices[:500]
selected_data = ocr_ds_split_0["split_0"].select(selected_indices)

# Convert to prompt/completion format
formatted_data = []
for item in selected_data:
    formatted_item = {
        "prompt": item["input"],
        "completion": item["output"]
    }
    formatted_data.append(formatted_item)

# Shuffle again for good measure
random.shuffle(formatted_data)

# Split data 7:2:1
train_size = int(0.7 * len(formatted_data))
valid_size = int(0.2 * len(formatted_data))
# test_size will be the remainder

train_data = formatted_data[:train_size]
valid_data = formatted_data[train_size:train_size+valid_size]
test_data = formatted_data[train_size+valid_size:]

print(f"Splitting data: {len(train_data)} train, {len(valid_data)} validation, {len(test_data)} test examples")

# Save files in JSONL format
def save_jsonl(data, filename):
    with open(os.path.join("jsonl-dataset", filename), 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

# Save the files
save_jsonl(train_data, "Train.jsonl")
save_jsonl(valid_data, "Valid.jsonl")
save_jsonl(test_data, "Test.jsonl")

print("Data saved to Train.jsonl, Valid.jsonl, and Test.jsonl")