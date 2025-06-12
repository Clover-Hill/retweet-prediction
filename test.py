import numpy as np
import datasets 
from transformers import AutoModel

dataset = datasets.load_from_disk("/fs-computility/plm/shared/jqcao/projects/retweet-prediction/data/preprocessed_dataset")["train"]

def text_enrich(sample):
    ret = ""
    ret += "Text Length: {} | ".format(sample['text_length'])
    ret += "Low Time Interval: {} | ".format(sample['low_time_interval'])
    ret += "Followers: {} | ".format(sample['user_followers_count'])
    ret += "URLs: {} | ".format(sample['urls'] if sample.get('urls') else 0)
    ret += "Hashtags: {} | ".format(sample['hashtags'] if sample.get('hashtags') else 0)
    ret += "Tweet: {}".format(sample['text'])
    
    return ret

for i in range(10):
    print(text_enrich(dataset[i]))