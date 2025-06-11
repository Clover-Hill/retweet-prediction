from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error
import numpy as np
import torch
from tqdm import tqdm

def evaluation_loop(model, eval_dataloader, head_type, accelerator):
    """
    Evaluate the model on the evaluation dataset.
    
    Args:
        model: The model to evaluate
        eval_dataloader: DataLoader for evaluation data
        accelerator: Accelerator object
        head_type: 'regression' or 'classification'
    
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
            # Get appropriate labels based on head type
            if head_type == "regression":
                labels = batch['retweet_counts']
            else:  # classification
                labels = batch['if_viral']
            
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
                # Convert logits to predictions for classification
                predictions = (torch.sigmoid(gathered_logits) > 0.5).float()
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
        
    else:  # classification
        # Convert to binary predictions
        all_predictions = [int(p) for p in all_predictions]
        all_labels = [int(l) for l in all_labels]
        
        # Classification metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, zero_division=0)
        recall = recall_score(all_labels, all_predictions, zero_division=0)
        f1 = f1_score(all_labels, all_predictions, zero_division=0)
        
        metrics['accuracy'] = accuracy
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        
    return metrics