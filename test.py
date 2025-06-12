import numpy as np
import datasets 
from transformers import AutoModel
from tqdm import tqdm

# Load all splits
dataset = datasets.load_from_disk("/fs-computility/plm/shared/jqcao/projects/retweet-prediction/data/feature_dataset")

import pdb 
pdb.set_trace()