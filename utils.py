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

# 1. Modified Feature Collator with Viral Class Support
def feature_collator(batch, tokenizer, intervals=None, max_length=512):
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
    scalar_features = F.normalize(scalar_features, p=2, dim=-1)
    
    # Process labels
    retweet_counts = torch.tensor([item['retweet_count'] for item in batch], dtype=torch.float32)
    if_viral = torch.tensor([item['if_viral'] for item in batch], dtype=torch.float32)
    
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
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
        'scalar_features': scalar_features,
        'retweet_counts': retweet_counts,
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
def evaluation_loop(model, eval_dataloader, head_type, accelerator, num_classes=16):
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
    all_labels = []
    all_logits = []  # For multi-class, we might want probabilities
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process):
            # Get appropriate labels based on head type
            if head_type == "regression":
                labels = batch['retweet_counts']
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
                all_logits.extend(probabilities.cpu().numpy())
            else:
                # For regression, predictions are the logits themselves
                predictions = gathered_logits
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(gathered_labels.cpu().numpy())
    
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
        # Multi-class classification metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Overall accuracy
        accuracy = accuracy_score(all_labels, all_predictions)
        metrics['accuracy'] = accuracy
        
        # If we have probabilities, calculate additional metrics
        if all_logits:
            all_probs = np.array(all_logits)
            
            # Top-k accuracy
            for k in [3, 5]:
                if k < num_classes:
                    top_k_preds = np.argsort(all_probs, axis=1)[:, -k:]
                    top_k_correct = np.any(top_k_preds == all_labels[:, np.newaxis], axis=1)
                    top_k_accuracy = np.mean(top_k_correct)
                    metrics[f'top_{k}_accuracy'] = top_k_accuracy
            
            # Average confidence in predictions
            max_probs = np.max(all_probs, axis=1)
            metrics['mean_confidence'] = np.mean(max_probs)
            metrics['std_confidence'] = np.std(max_probs)
            
            # Log loss (cross-entropy)
            epsilon = 1e-15  # For numerical stability
            all_probs = np.clip(all_probs, epsilon, 1 - epsilon)
            log_loss_value = -np.mean(np.log(all_probs[np.arange(len(all_labels)), all_labels]))
            metrics['log_loss'] = log_loss_value
        
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
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        metrics['confusion_matrix'] = cm.tolist()  # Convert to list for JSON serialization
        
        # Per-class metrics
        per_class_precision = precision_score(all_labels, all_predictions, average=None, zero_division=0)
        per_class_recall = recall_score(all_labels, all_predictions, average=None, zero_division=0)
        per_class_f1 = f1_score(all_labels, all_predictions, average=None, zero_division=0)
        
        # Store per-class metrics
        for i in range(num_classes):
            metrics[f'class_{i}_precision'] = per_class_precision[i] if i < len(per_class_precision) else 0.0
            metrics[f'class_{i}_recall'] = per_class_recall[i] if i < len(per_class_recall) else 0.0
            metrics[f'class_{i}_f1'] = per_class_f1[i] if i < len(per_class_f1) else 0.0
        
    return metrics