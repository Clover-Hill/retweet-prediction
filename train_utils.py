import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import KFold
from tqdm import tqdm
from typing import Optional, Dict, List
import json

# Define feature categories based on the generated dataset
DENSE_FEATURES = [
    # Base metrics transformations
    'user_followers_count_log', 'user_followers_count_z', 'user_followers_count_rank', 'user_followers_count_cdf',
    'user_friends_count_log', 'user_friends_count_z', 'user_friends_count_rank', 'user_friends_count_cdf',
    'user_statuses_count_log', 'user_statuses_count_z', 'user_statuses_count_rank', 'user_statuses_count_cdf',
    'followers_friends_log', 'followers_friends_z', 'followers_friends_rank', 'followers_friends_cdf',
    'followers_statuses_log', 'followers_statuses_z', 'followers_statuses_rank', 'followers_statuses_cdf',
    'friends_statuses_log', 'friends_statuses_z', 'friends_statuses_rank', 'friends_statuses_cdf',
    'followers_friends_statuses_log', 'followers_friends_statuses_z', 'followers_friends_statuses_rank', 'followers_friends_statuses_cdf',
    # Count features
    'mentions_count', 'hashtags_count', 'urls_count', 'n_unique_domains',
    # Text features
    'text_length', 'word_count', 'n_capital_letters', 'n_exclamation', 'n_question', 'n_punctuation',
    'capital_ratio', 'exclamation_ratio', 'punctuation_ratio',
    # TF-IDF features
    'tfidf_svd_0', 'tfidf_svd_1', 'tfidf_svd_2', 'tfidf_svd_3', 'tfidf_svd_4',
    # Encoding features
    'weekday_count_enc', 'hour_count_enc', 'day_count_enc', 'month_count_enc', 
    'user_verified_count_enc', 'user_cluster_count_enc',
    'weekday_target_enc', 'hour_target_enc', 'user_cluster_target_enc',
    'user_followers_count_qbin_10_target_enc', 'user_friends_count_qbin_10_target_enc',
    # Time features
    'hours_from_latest',
    # Original metrics (for compatibility)
    'user_followers_count', 'user_friends_count', 'user_statuses_count',
    'followers_friends', 'followers_statuses', 'friends_statuses', 'followers_friends_statuses'
]

SPARSE_FEATURES = [
    'user_verified', 'weekday', 'hour', 'day', 'month', 'week_of_month',
    'user_cluster',
    'user_followers_count_qbin_10', 'user_friends_count_qbin_10', 'user_statuses_count_qbin_10',
    'followers_friends_qbin_10', 'followers_statuses_qbin_10', 'friends_statuses_qbin_10',
    'followers_friends_statuses_qbin_10',
    'has_mentions', 'has_hashtags', 'has_urls', 'has_https', 'has_url_shortener'
]

VARLEN_SPARSE_FEATURES = [
    'user_mentions', 'urls', 'hashtags'
]

def custom_pad_sequences(sequences, maxlen=None, padding='post', value=0):
    """
    Custom implementation of pad_sequences using PyTorch
    
    Args:
        sequences: List of sequences (lists) to pad
        maxlen: Maximum length. If None, uses the length of the longest sequence
        padding: 'pre' or 'post' - where to add padding
        value: Value to use for padding
    
    Returns:
        Numpy array of padded sequences
    """
    if not sequences:
        return np.array([])
    
    # Find the maximum length if not specified
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)
    
    # Create padded array
    padded = np.full((len(sequences), maxlen), value, dtype=np.int64)
    
    for i, seq in enumerate(sequences):
        seq = np.array(seq)
        if len(seq) > maxlen:
            # Truncate if sequence is too long
            if padding == 'post':
                padded[i] = seq[:maxlen]
            else:  # 'pre'
                padded[i] = seq[-maxlen:]
        else:
            # Pad if sequence is too short
            if padding == 'post':
                padded[i, :len(seq)] = seq
            else:  # 'pre'
                padded[i, -len(seq):] = seq
    
    return padded

def get_feature_dimensions(dataset):
    """
    Calculate the dimensions for each feature type based on the dataset
    
    Args:
        dataset: HuggingFace dataset or pandas DataFrame
    
    Returns:
        Tuple of (dense_features_dim, sparse_feature_dims, varlen_feature_dims)
    """
    # Dense features dimension is simply the count
    dense_features_dim = len(DENSE_FEATURES)
    
    # For sparse features, we need to count unique values
    sparse_feature_dims = {}
    
    # Convert to pandas if it's a HF dataset
    if hasattr(dataset, 'to_pandas'):
        df = dataset.to_pandas()
    else:
        df = dataset
    
    for feat in SPARSE_FEATURES:
        if feat in df.columns:
            # Count unique values + 1 for unknown/padding
            n_unique = df[feat].nunique()
            # Add extra space for unknown values
            sparse_feature_dims[feat] = n_unique + 2
        else:
            print(f"Warning: Feature {feat} not found in dataset")
            sparse_feature_dims[feat] = 100  # Default value
    
    # For variable-length features, we need vocabulary size
    varlen_feature_dims = {}
    
    for feat in VARLEN_SPARSE_FEATURES:
        if feat in df.columns:
            # Collect all unique items across all rows
            vocab = set()
            for items in df[feat].dropna():
                if isinstance(items, str) and items:
                    vocab.update(items.split(','))
            
            # Vocabulary size + padding + unknown
            varlen_feature_dims[feat] = len(vocab) + 2
            # Cap at reasonable maximum
            varlen_feature_dims[feat] = min(varlen_feature_dims[feat], 50000)
        else:
            print(f"Warning: Feature {feat} not found in dataset")
            varlen_feature_dims[feat] = 10000  # Default value
    
    return dense_features_dim, sparse_feature_dims, varlen_feature_dims

def create_vocabulary_mapping(dataset, varlen_features=VARLEN_SPARSE_FEATURES):
    """
    Create vocabulary mappings for variable-length features
    
    Args:
        dataset: HuggingFace dataset or pandas DataFrame
        varlen_features: List of variable-length feature names
    
    Returns:
        Dict of feature_name -> {token -> index} mappings
    """
    vocab_mappings = {}
    
    # Convert to pandas if needed
    if hasattr(dataset, 'to_pandas'):
        df = dataset.to_pandas()
    else:
        df = dataset
    
    for feat in varlen_features:
        if feat in df.columns:
            vocab = {}
            vocab['<PAD>'] = 0
            vocab['<UNK>'] = 1
            idx = 2
            
            # Collect vocabulary
            token_counts = {}
            for items in df[feat].dropna():
                if isinstance(items, str) and items:
                    for token in items.split(','):
                        token = token.strip()
                        if token:
                            token_counts[token] = token_counts.get(token, 0) + 1
            
            # Sort by frequency and add to vocabulary
            for token, _ in sorted(token_counts.items(), key=lambda x: x[1], reverse=True):
                if idx < 50000:  # Cap vocabulary size
                    vocab[token] = idx
                    idx += 1
            
            vocab_mappings[feat] = vocab
    
    return vocab_mappings

def process_varlen_feature_with_vocab(feature_str, vocab, max_len=20):
    """Process variable length feature strings into padded sequences using vocabulary"""
    if pd.isna(feature_str) or not feature_str:
        return [0]  # Return padding token
    
    items = feature_str.split(',')
    indices = []
    
    for item in items[:max_len]:
        item = item.strip()
        if item in vocab:
            indices.append(vocab[item])
        else:
            indices.append(vocab.get('<UNK>', 1))
    
    return indices

def feature_collator(batch, tokenizer=None, use_text=True, max_length=512, 
                    varlen_max_len=20, vocab_mappings=None):
    """
    Collate function for both text and feature-based models
    
    Args:
        batch: List of dictionaries containing raw data
        tokenizer: Tokenizer for text processing (None for feature-only model)
        use_text: Whether to process text for Qwen3 model
        max_length: Maximum sequence length for tokenization
        varlen_max_len: Maximum length for variable-length features
        vocab_mappings: Dictionary of vocabularies for variable-length features
    
    Returns:
        Dictionary with model inputs
    """
    batch_size = len(batch)
    
    # Process dense features
    dense_features_tensor = torch.zeros(batch_size, len(DENSE_FEATURES), dtype=torch.float32)
    for i, item in enumerate(batch):
        for j, feat in enumerate(DENSE_FEATURES):
            value = item.get(feat, 0)
            if pd.isna(value):
                value = 0
            dense_features_tensor[i, j] = float(value)
    
    # Process sparse features
    sparse_features_tensor = torch.zeros(batch_size, len(SPARSE_FEATURES), dtype=torch.long)
    for i, item in enumerate(batch):
        for j, feat in enumerate(SPARSE_FEATURES):
            value = item.get(feat, 0)
            if pd.isna(value):
                value = 0
            sparse_features_tensor[i, j] = int(value)
    
    # Process variable-length sparse features
    varlen_features_dict = {}
    for feat in VARLEN_SPARSE_FEATURES:
        sequences = []
        for item in batch:
            if vocab_mappings and feat in vocab_mappings:
                seq = process_varlen_feature_with_vocab(
                    item.get(feat, ''), 
                    vocab_mappings[feat], 
                    varlen_max_len
                )
            else:
                raise ValueError(f"Vocabulary mapping for {feat} not provided")
            sequences.append(seq)
        # Pad sequences
        padded = custom_pad_sequences(sequences, maxlen=varlen_max_len, padding='post', value=0)
        varlen_features_dict[feat] = torch.tensor(padded, dtype=torch.long)
    
    # Process labels
    if batch[0].get('retweet_count') is None:
        labels = None
    else:
        labels = torch.tensor([item['retweet_count'] for item in batch], dtype=torch.float32)
    
    result = {
        'dense_features': dense_features_tensor,
        'sparse_features': sparse_features_tensor,
        'varlen_features': varlen_features_dict,
        'labels': labels
    }
    
    # Add text processing if needed
    if use_text and tokenizer is not None:
        texts = [item.get('text', '') for item in batch]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        result['input_ids'] = encoded['input_ids']
        result['attention_mask'] = encoded['attention_mask']
    
    return result

def evaluation_loop(model, eval_dataloader, accelerator, use_text=True):
    """
    Evaluate the model on the evaluation dataset.
    
    Args:
        model: The model to evaluate
        eval_dataloader: DataLoader for evaluation data
        accelerator: Accelerator object
        use_text: Whether the model uses text inputs
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process):
            labels = batch['labels']
            
            # Forward pass
            if use_text:
                outputs = model(
                    input_ids=batch.get('input_ids'),
                    attention_mask=batch.get('attention_mask'),
                    dense_features=batch['dense_features'],
                    sparse_features=batch['sparse_features'],
                    varlen_features=batch['varlen_features'],
                    labels=labels
                )
            else:
                outputs = model(
                    dense_features=batch['dense_features'],
                    sparse_features=batch['sparse_features'],
                    varlen_features=batch['varlen_features'],
                    labels=labels
                )
            
            loss = outputs.loss
            logits = outputs.logits
            
            # Gather predictions and labels from all processes
            gathered_loss = accelerator.gather(loss)
            gathered_logits = accelerator.gather(logits)
            gathered_labels = accelerator.gather(labels)
            
            total_loss += gathered_loss.mean().item()
            num_batches += 1
            
            all_predictions.extend(gathered_logits.cpu().numpy())
            all_labels.extend(gathered_labels.cpu().numpy())
    
    # Calculate metrics
    metrics = {}
    avg_loss = total_loss / num_batches
    metrics['eval_loss'] = avg_loss
    
    # Regression metrics
    mae = mean_absolute_error(all_labels, all_predictions)
    mse = mean_squared_error(all_labels, all_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_labels, all_predictions)
    
    # Log-based metrics (for retweet counts)
    all_labels_log = np.log1p(all_labels)
    all_predictions_log = np.log1p(np.maximum(all_predictions, 0))
    msle = mean_squared_error(all_labels_log, all_predictions_log)
    
    metrics.update({
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'msle': msle
    })
    
    return metrics

def label_scaling(y):
    """Apply log transformation and min-max scaling to labels"""
    y_log = np.log1p(y)
    y_min = y_log.min()
    y_max = y_log.max()
    y_scaled = (y_log - y_min) / (y_max - y_min + 1e-8)
    
    scaler_info = {'min': y_min, 'max': y_max}
    return scaler_info, y_scaled

def label_inverse_scaling(scaler_info, y_scaled):
    """Inverse the label scaling"""
    y_log = y_scaled * (scaler_info['max'] - scaler_info['min']) + scaler_info['min']
    y = np.expm1(y_log)
    return y