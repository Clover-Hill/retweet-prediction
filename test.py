import numpy as np
import datasets 
from transformers import AutoModel

dataset = datasets.load_from_disk("/fs-computility/plm/shared/jqcao/projects/retweet-prediction/data/preprocessed_dataset")["eval"]
dataset = dataset.filter(lambda x: x["retweet_count"] >= 10)

# print mean of retweet_count
mean_retweet_count = np.mean(dataset["retweet_count"])
print(f"Mean retweet count: {mean_retweet_count}")