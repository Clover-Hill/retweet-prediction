import datasets
import pandas as pd
import numpy as np
from datetime import datetime

# Configuration
embedding_path = "/home/jqcao/projects/retweet-prediction/data/embedding_dataset"
output_path = "/home/jqcao/projects/retweet-prediction/data/preprocessed_dataset"

# Load dataset
print(f"Loading dataset from: {embedding_path}")
embedding_dataset = datasets.load_from_disk(embedding_path)

def preprocess_split(dataset, split_name):
    """Preprocess a single dataset split"""
    print(f"\nProcessing {split_name} split...")
    
    # Get timestamp range for interval calculation
    timestamps = dataset['timestamp']
    earliest_timestamp = min(timestamps)
    latest_timestamp = max(timestamps)
    time_range = latest_timestamp - earliest_timestamp
    interval_size = time_range / 20
    
    def process_example(example):
        # Calculate time interval (1-20)
        timestamp_ms = example['timestamp']
        interval = int((timestamp_ms - earliest_timestamp) / interval_size) + 1
        interval = min(interval, 20)
        
        # Get hour from timestamp
        dt = pd.to_datetime(timestamp_ms, unit='ms')
        hour = dt.hour
        
        # Create processed example
        processed = {
            # Direct copies
            'id': example['id'],
            'retweet_count': example['retweet_count'],
            'text': example['text'],
            # 'text_embedding': example['embeddings'],
            'user_verified': example['user_verified'],
            'user_followers_count': example['user_followers_count'],
            
            # Boolean features
            'has_url': example.get('urls') is not None and len(example.get('urls', [])) > 0,
            'has_hashtags': example.get('hashtags') is not None and len(example.get('hashtags', [])) > 0,
            
            # Computed features
            'text_length': len(example['text']) if example['text'] else 0,
            'low_time_interval': interval in [5, 6, 7, 8],  # Low viral intervals
            'high_hour_interval': 6 <= hour < 12,  # Morning hours (high viral)
        }
        
        # Add viral flag for train/eval only
        if split_name in ['train', 'eval']:
            processed['if_viral'] = example['retweet_count'] >= 10
        
        return processed
    
    # Process dataset
    processed = dataset.map(
        process_example,
        batched=False,
        num_proc=32,
        load_from_cache_file=False,
        desc=f"Processing {split_name}"
    )
    
    # Select columns
    columns = [
        'text', 'retweet_count', 'has_url', 'has_hashtags', 
        'user_verified', 'text_length', 'user_followers_count',
        'low_time_interval', 'high_hour_interval'
    ]
    if split_name in ['train', 'eval']:
        columns.append('if_viral')
    
    return processed.select_columns(columns)

# Process all splits
processed_splits = {}
for split_name in embedding_dataset.keys():
    processed_splits[split_name] = preprocess_split(embedding_dataset[split_name], split_name)

# Create and save new dataset
preprocessed_dataset = datasets.DatasetDict(processed_splits)
print(f"\nSaving to: {output_path}")
preprocessed_dataset.save_to_disk(output_path)

# Print summary
print("\n" + "="*60)
print("PREPROCESSING COMPLETE")
print("="*60)

for split_name, split_data in preprocessed_dataset.items():
    print(f"\n{split_name.upper()} Split: {len(split_data):,} examples")
    
    # Quick stats
    df = split_data.to_pandas()
    print(f"  - Has URL: {df['has_url'].mean()*100:.1f}%")
    print(f"  - Has Hashtags: {df['has_hashtags'].mean()*100:.1f}%")
    print(f"  - Low Time Interval: {df['low_time_interval'].mean()*100:.1f}%")
    print(f"  - High Hour Interval: {df['high_hour_interval'].mean()*100:.1f}%")
    if 'if_viral' in df.columns:
        print(f"  - Viral Rate: {df['if_viral'].mean()*100:.1f}%")
        
# ============================================================
# PREPROCESSING COMPLETE
# ============================================================

# TRAIN Split: 662,777 examples
#   - Has URL: 32.2%
#   - Has Hashtags: 10.9%
#   - Low Time Interval: 11.6%
#   - High Hour Interval: 19.1%
#   - Viral Rate: 15.1%

# TEST Split: 285,334 examples
#   - Has URL: 32.3%
#   - Has Hashtags: 11.0%
#   - Low Time Interval: 11.6%
#   - High Hour Interval: 19.0%

# EVAL Split: 3,000 examples
#   - Has URL: 30.5%
#   - Has Hashtags: 10.9%
#   - Low Time Interval: 10.9%
#   - High Hour Interval: 18.3%
#   - Viral Rate: 15.5%