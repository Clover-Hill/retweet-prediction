import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, confusion_matrix,
    classification_report
)
from tqdm import tqdm
from typing import Optional, Tuple
from transformers.modeling_outputs import SequenceClassifierOutput

def text_enrich(sample):
    ret = ""
    ret += "Text Length: {} | ".format(sample['text_length'])
    ret += "Low Time Interval: {} | ".format(sample['low_time_interval'])
    ret += "Followers: {} | ".format(sample['user_followers_count'])
    ret += "URLs: {} | ".format(sample['urls'] if sample.get('urls') else 0)
    ret += "Hashtags: {} | ".format(sample['hashtags'] if sample.get('hashtags') else 0)
    ret += "Tweet: {}".format(sample['text'])
    
    return ret

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

# 1. Modified Feature Collator with Viral Class Support
def feature_collator(batch, tokenizer, intervals=None, max_length=512, use_rich_text=False):
    """
    Collate function to process batch data with log1p for heavy-tailed features
    and L2 normalization. Now supports viral_class labels.
    
    Args:
        batch: List of dictionaries containing raw data
        tokenizer: Tokenizer for text processing
        intervals_path: Path to count_intervals.json file
        max_length: Maximum sequence length for tokenization
    
    Returns:
        Dictionary with model inputs
    """
    
    # Extract texts and tokenize
    if use_rich_text:
        texts = [text_enrich(item) for item in batch]
    else:
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
    
    # Process dense features
    scalar_features = torch.zeros(batch_size, len(DENSE_FEATURES), dtype=torch.float32)
    for i, item in enumerate(batch):
        for j, feat in enumerate(DENSE_FEATURES):
            value = item.get(feat, 0)
            scalar_features[i, j] = float(value)
    
    # Apply L2 normalization to scalar features
    scalar_features = F.normalize(scalar_features, p=2, dim=-1)
    
    # Process labels
    if batch[0]['retweet_count'] is None:
        retweet_count = torch.tensor([-1] * batch_size, dtype=torch.float32)
        if_viral = torch.tensor([-1] * batch_size, dtype=torch.float32)
    else:
        retweet_count = torch.tensor([item['retweet_count'] for item in batch], dtype=torch.float32)
        if_viral = torch.tensor([float(item['retweet_count']>=10) for item in batch], dtype=torch.float32)
    
    id = torch.tensor([item['id'] for item in batch], dtype=torch.long)
    
    # Calculate viral_class if intervals provided
    if intervals is not None:
        viral_class = []
        for item in batch:
            count = item['retweet_count']
            class_idx = get_class_for_retweet_count(count, intervals)
            assert class_idx is not None, f"Class index not found for retweet count {count}"
            viral_class.append(class_idx if class_idx is not None else 0)  # Default to 0 if not found
        viral_class = torch.tensor(viral_class, dtype=torch.long)
    
    result = {
        'id': id,
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
        'scalar_features': scalar_features,
        'retweet_count': retweet_count,
        'if_viral': if_viral
    }
    
    if intervals is not None:
        result['viral_class'] = viral_class
    
    return result

def get_class_for_retweet_count(retweet_count, intervals):
    # for non-viral retweet counts, we return class -1
    if retweet_count < 10:
        return -1
    """Find which class a retweet count belongs to"""
    for interval in intervals:
        if interval['interval_start'] <= retweet_count < interval['interval_end']:
            return interval['class_index']
    # Handle edge case for the last interval
    if retweet_count >= intervals[-1]['interval_start']:
        return intervals[-1]['class_index']
    return None

# 2. Modified Evaluation Loop with Multi-class Support
def evaluation_loop(model, eval_dataloader, head_type, accelerator, num_classes=16, intervals=None):
    """
    Evaluate the model on the evaluation dataset.
    
    Args:
        model: The model to evaluate
        eval_dataloader: DataLoader for evaluation data
        head_type: 'regression', 'classification', or 'multi_class'
        accelerator: Accelerator object
        num_classes: Number of classes for multi-class classification
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_retweet_count = []
    all_labels = []
    all_logits = []  # For multi-class, we might want probabilities
    all_weighted_predictions = []
    total_loss = 0
    num_batches = 0
    
    if intervals is not None:
        weight = torch.tensor([interval['mean_val'] for interval in intervals], dtype=torch.float32)
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process):
            # Get appropriate labels based on head type
            if head_type == "regression":
                labels = batch['retweet_count']
            elif head_type == "classification":
                labels = batch['if_viral']
            elif head_type == "multi_regression":  # multi_class
                labels = batch['viral_class']
            
            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                scalar_features=batch['scalar_features'],
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            # Gather predictions and labels from all processes
            gathered_loss = accelerator.gather(loss)
            gathered_logits = accelerator.gather(logits)
            gathered_labels = accelerator.gather(labels)
            gathered_retweet_count = accelerator.gather(batch['retweet_count'])
            
            total_loss += gathered_loss.mean().item()
            num_batches += 1
            
            if head_type == "classification":
                # Convert logits to predictions for binary classification
                predictions = (torch.sigmoid(gathered_logits) > 0.5).float()
            elif head_type == "multi_regression":
                # For multi-class, get the class with highest probability
                predictions = torch.argmax(gathered_logits, dim=-1)
                # Store probabilities instead of raw logits
                probabilities = torch.softmax(gathered_logits, dim=-1)

                weighted_predictions = torch.sum(probabilities * weight.to(probabilities.device), dim=-1)
                all_weighted_predictions.extend(weighted_predictions.cpu().numpy())
                all_logits.extend(probabilities.cpu().numpy())
            else:
                # For regression, predictions are the logits themselves
                predictions = torch.exp(gathered_logits)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(gathered_labels.cpu().numpy())
            all_retweet_count.extend(gathered_retweet_count.cpu().numpy())
    
    # Calculate metrics based on head type
    metrics = {}
    avg_loss = total_loss / num_batches
    metrics['eval_loss'] = avg_loss
    
    if head_type == "regression":
        # Regression metrics
        mae = mean_absolute_error(all_labels, all_predictions)
        mse = mean_squared_error(all_labels, all_predictions)
        rmse = np.sqrt(mse)
        
        metrics['mae'] = mae
        metrics['mse'] = mse
        metrics['rmse'] = rmse
        
        # Calculate R-squared
        ss_res = np.sum((np.array(all_labels) - np.array(all_predictions)) ** 2)
        ss_tot = np.sum((np.array(all_labels) - np.mean(all_labels)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        metrics['r2'] = r2
        
    elif head_type == "classification":
        # Binary classification metrics
        all_predictions = [int(p) for p in all_predictions]
        all_labels = [int(l) for l in all_labels]
        
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, zero_division=0)
        recall = recall_score(all_labels, all_predictions, zero_division=0)
        f1 = f1_score(all_labels, all_predictions, zero_division=0)
        
        metrics['accuracy'] = accuracy
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        
    elif head_type == "multi_regression":  # multi_class
        mae = mean_absolute_error(all_retweet_count, all_weighted_predictions)
        mse = mean_squared_error(all_retweet_count, all_weighted_predictions)
        
        metrics['mae'] = mae
        metrics['mse'] = mse

        # Multi-class classification metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Overall accuracy
        accuracy = accuracy_score(all_labels, all_predictions)
        metrics['accuracy'] = accuracy
        
        # Per-class and averaged metrics
        precision_macro = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
        recall_macro = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
        f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        
        precision_micro = precision_score(all_labels, all_predictions, average='micro', zero_division=0)
        recall_micro = recall_score(all_labels, all_predictions, average='micro', zero_division=0)
        f1_micro = f1_score(all_labels, all_predictions, average='micro', zero_division=0)
        
        metrics['precision_macro'] = precision_macro
        metrics['recall_macro'] = recall_macro
        metrics['f1_macro'] = f1_macro
        metrics['precision_micro'] = precision_micro
        metrics['recall_micro'] = recall_micro
        metrics['f1_micro'] = f1_micro
        
        
    return metrics