# 📈 Retweet Prediction with Deep Regression Models

Predicting retweet counts with powerful language models and deep learning pipelines.

## 🧠 Overview

This project builds a regression model to predict retweet counts from tweet embeddings using advanced language models and multi-layer perceptrons (MLPs). It leverages [Qwen3 Embeddings](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), deep regression heads, and the [🤗 Accelerate](https://huggingface.co/docs/accelerate/index) library for efficient distributed training.

---

## 🗂️ Project Structure

```
📁 accelerate_config        # Accelerate configurations for training/evaluation
📁 checkpoint               # Saved models/checkpoints
📁 data                     # Raw and processed datasets
📁 feature_engineering      # (Optional) Extra preprocessing or custom features
📁 imgs                     # Visuals (e.g., architecture diagrams)
📁 model                    # Model definition files
📁 result                   # Evaluation/test results
📁 scripts                  # Script utilities or pipelines
📁 utils                    # Dataset and feature utility scripts
    └── create_dataset.py
    └── generate_feature.py
    └── generate_text_embedding.py
    └── preprocess_dataset.py
📁 wandb                    # Weights & Biases tracking logs
📄 train.py                 # Main training entry point
📄 train_dnn.py             # Alternate training with different DNN configurations
📄 train_utils.py           # Training helper functions
📄 train_utils_dnn.py       # DNN-specific training helpers
📄 eval.py                  # General evaluation script
📄 eval_regres.py           # Regression-specific evaluation
📄 test.py                  # Basic testing script
📄 README.md                # This file
📄 .gitignore               # Git ignore file
```

---

## 🚀 Training

Use `train.script` to launch training with HuggingFace Accelerate:

```bash
./scripts/train_{method}.sh
```

### 🔧 Key Training Options

- `--model_name_or_path`: Path to pretrained embedding model (e.g. Qwen3)
- `--dataset_name`: Path to preprocessed feature dataset
- `--num_train_epochs`: Number of training epochs
- `--head_type`: Prediction head type (`regression`)
- `--mlp_num`: Number of MLP layers
- `--dropout_rate`: Dropout rate
- `--output_dir`: Where to save model checkpoints

---

## 📊 Evaluation

Evaluate trained regression models using:

```bash
./scripts/eval_{method}.sh
```

### 🔍 Evaluation Parameters

- `--regression_model_path`: Path to a trained model checkpoint
- `--dataset_name`: Dataset for testing
- `--do_test`: Flag to run test set evaluation
- `--output_dir`: Where to save results

---

## 📦 Requirements

- Python 3.8+
- PyTorch
- Transformers
- Accelerate
- Datasets
- wandb (optional for logging)

---

## ✨ Highlights

- ⚡ Distributed training via HuggingFace Accelerate
- 🔤 Rich tweet embeddings via large language models
- 🔁 Modular and reusable preprocessing utilities
- 📉 Regression head designed for retweet prediction
- 🧪 Easy evaluation and testing pipelines

---