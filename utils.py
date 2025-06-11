import torch
import numpy as np
import torch.nn.functional as F

def feature_collator(batch, tokenizer, max_length=512):
    """
    Collate function to process batch data with log1p for heavy-tailed features
    and L2 normalization.
    
    Args:
        batch: List of dictionaries containing raw data
        tokenizer: Tokenizer for text processing
        max_length: Maximum sequence length for tokenization
    
    Returns:
        Dictionary with model inputs
    """
    # Extract texts and tokenize
    texts = [item['text'] for item in batch]
    
    # Tokenize texts
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Process scalar features
    batch_size = len(batch)
    scalar_features = torch.zeros(batch_size, 7, dtype=torch.float32)
    
    # Feature indices mapping
    feature_mapping = {
        'has_url': 0,
        'has_hashtags': 1, 
        'user_verified': 2,
        'text_length': 3,
        'user_followers_count': 4,
        'low_time_interval': 5,
        'high_hour_interval': 6
    }
    
    # Fill scalar features
    for i, item in enumerate(batch):
        # Boolean features (keep as 0 or 1)
        scalar_features[i, feature_mapping['has_url']] = float(item['has_url'])
        scalar_features[i, feature_mapping['has_hashtags']] = float(item['has_hashtags'])
        scalar_features[i, feature_mapping['user_verified']] = float(item['user_verified'])
        scalar_features[i, feature_mapping['low_time_interval']] = float(item['low_time_interval'])
        scalar_features[i, feature_mapping['high_hour_interval']] = float(item['high_hour_interval'])
        
        # Numerical features with log1p transformation for heavy-tailed distributions
        scalar_features[i, feature_mapping['text_length']] = np.log1p(item['text_length'])
        scalar_features[i, feature_mapping['user_followers_count']] = np.log1p(item['user_followers_count'])
    
    # Apply L2 normalization to scalar features
    # This ensures all features contribute equally regardless of scale
    scalar_features = F.normalize(scalar_features, p=2, dim=-1)
    
    # Process labels
    retweet_counts = torch.tensor([item['retweet_count'] for item in batch], dtype=torch.float32)
    if_viral = torch.tensor([item['if_viral'] for item in batch], dtype=torch.float32)
    
    return {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
        'scalar_features': scalar_features,
        'retweet_counts': retweet_counts,
        'if_viral': if_viral
    }