import datasets
import numpy as np
import json
import pickle

# Load and filter dataset
dataset = datasets.load_from_disk("/fs-computility/plm/shared/jqcao/projects/retweet-prediction/data/preprocessed_dataset")["train"]
dataset = dataset.filter(lambda x: x["retweet_count"] >= 10)
retweet_counts = dataset["retweet_count"]

class_num = 16

# Convert to numpy array for easier manipulation
data = np.array(retweet_counts)

# Sort the data
sorted_data = np.sort(data)

# Calculate samples per class
total_samples = len(sorted_data)
samples_per_class = total_samples // class_num
remainder = total_samples % class_num

# Create class intervals
class_info = []

start_idx = 0
for i in range(class_num):
    # Distribute remainder samples across first few classes
    if i < remainder:
        end_idx = start_idx + samples_per_class + 1
    else:
        end_idx = start_idx + samples_per_class
    
    # Get samples for this class
    class_samples = sorted_data[start_idx:end_idx]
    
    # Calculate statistics
    if len(class_samples) > 0:
        class_info.append({
            "class_index": i,
            "interval_start": float(class_samples[0]),  # Min value in class
            "interval_end": float(class_samples[-1]),   # Max value in class
            "mean_val": float(np.mean(class_samples)),
            "num_samples": len(class_samples),
            "sample_indices": (start_idx, end_idx)  # For reference
        })
    
    start_idx = end_idx

# Print summary statistics
print(f"Total samples: {total_samples}")
print(f"Samples per class (base): {samples_per_class}")
print(f"Classes with extra sample: {remainder}")
for i in range(class_num):
    cls = class_info[i]
    print(f"  Class {i}: [{cls['interval_start']:.2f}, {cls['interval_end']:.2f}], "
          f"mean={cls['mean_val']:.2f}, n={cls['num_samples']}")
    
# Save to JSON file
with open("feature_engineering/count_intervals.json", "w") as f:
    json.dump(class_info, f, indent=2)