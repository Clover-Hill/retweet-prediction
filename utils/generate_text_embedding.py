import numpy as np
import datasets
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch import Tensor

# Load the dataset
raw_dataset = datasets.load_from_disk("data/raw_dataset")

# ----------------------------------1. Add if_viral column to train and eval splits----------------------------------
def add_viral_column(example):
    """Add if_viral column based on retweet_count"""
    example['if_viral'] = example['retweet_count'] >= 10
    return example

# Apply to train and eval splits only
if 'train' in raw_dataset:
    raw_dataset['train'] = raw_dataset['train'].map(
        add_viral_column,
        batched=False,
        num_proc=64,
        load_from_cache_file=False,
        )
if 'eval' in raw_dataset:
    raw_dataset['eval'] = raw_dataset['eval'].map(
        add_viral_column,
        batched=False,
        num_proc=64,
        load_from_cache_file=False,
        )

# ----------------------------------2. Get sentence embeddings using Qwen embedding model----------------------------------

# Initialize the model and tokenizer
model_name = "/data0/jqcao/models/Qwen3/Qwen3-Embedding-0.6B"  # Alternative to Qwen3-Embedding-0.6B
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
model = AutoModel.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# Define pooling function based on reference code
def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool embeddings using last token method"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_embeddings(examples):
    """Generate embeddings for text column using last token pooling"""
    # Tokenize the texts
    batch_dict = tokenizer(
        examples['text'], 
        padding=True, 
        truncation=True, 
        max_length=512,
        return_tensors='pt'
    )
    
    # Move to device
    batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
    
    # Generate embeddings
    with torch.no_grad():
        outputs = model(**batch_dict)
        # Use last token pooling instead of mean pooling
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
    # Convert to numpy and add to examples
    examples['embeddings'] = embeddings.cpu().numpy().tolist()
    return examples

# Process in batches for efficiency
batch_size = 16

# Apply embedding generation to all splits
for split in raw_dataset.keys():
    print(f"Processing {split} split...")
    raw_dataset[split] = raw_dataset[split].map(
        get_embeddings,
        batched=True,
        batch_size=batch_size,
        load_from_cache_file=False,
        desc=f"Generating embeddings for {split}"
    )
    
# Save the processed dataset
raw_dataset.save_to_disk("data/embedding_dataset")

# Verify the results
print("\nDataset info after processing:")
for split in raw_dataset.keys():
    print(f"\n{split} split:")
    print(f"  Number of examples: {len(raw_dataset[split])}")
    print(f"  Columns: {raw_dataset[split].column_names}")
    
    if split in ['train', 'eval']:
        viral_count = sum(raw_dataset[split]['if_viral'])
        print(f"  Viral posts (retweet_count >= 10): {viral_count} ({viral_count/len(raw_dataset[split])*100:.2f}%)")
    
    # Check embedding dimension and normalization
    if 'embeddings' in raw_dataset[split].column_names:
        embedding_dim = len(raw_dataset[split][0]['embeddings'])
        print(f"  Embedding dimension: {embedding_dim}")
        
        # Verify embeddings are normalized (L2 norm should be ~1.0)
        sample_embedding = np.array(raw_dataset[split][0]['embeddings'])
        norm = np.linalg.norm(sample_embedding)
        print(f"  Sample embedding L2 norm: {norm:.6f} (should be ~1.0)")

# Optional: Example of computing similarity between embeddings
def compute_similarity(embedding1, embedding2):
    """Compute cosine similarity between two normalized embeddings"""
    # Since embeddings are already normalized, dot product equals cosine similarity
    return np.dot(embedding1, embedding2)

# Example usage
if len(raw_dataset['train']) >= 2:
    emb1 = np.array(raw_dataset['train'][0]['embeddings'])
    emb2 = np.array(raw_dataset['train'][1]['embeddings'])
    similarity = compute_similarity(emb1, emb2)
    print(f"\nExample similarity between first two embeddings: {similarity:.4f}")
    print(f"Text 1: {raw_dataset['train'][0]['text']}")
    print(f"Text 2: {raw_dataset['train'][1]['text']}")