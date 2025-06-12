import numpy as np
import datasets 
from transformers import AutoModel

dataset = datasets.load_from_disk("/fs-computility/plm/shared/jqcao/projects/retweet-prediction/data/feature_dataset")

print(dataset['train'][0])