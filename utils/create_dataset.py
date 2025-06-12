import numpy as np
import datasets 

train_path = "./data/train.csv"
test_path = "./data/evaluation.csv"

raw_dataset = datasets.load_dataset("csv", data_files={"train": train_path, "test": test_path})

# Select 100 samples as test set
np.random.seed(42)

total_samples = len(raw_dataset["train"])

all_indices = np.arange(total_samples)
eval_indices = np.random.choice(all_indices, size=3000, replace=False)
train_indices = np.array([i for i in all_indices if i not in eval_indices])

raw_dataset["eval"] = raw_dataset["train"].select(eval_indices)
raw_dataset["train"] = raw_dataset["train"].select(train_indices)

print(raw_dataset)
print(f"Train set size: {len(raw_dataset['train'])}, Eval set size: {len(raw_dataset['eval'])}, Test set size: {len(raw_dataset['test'])}")

raw_dataset.save_to_disk("./data/raw_dataset")