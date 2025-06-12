#!/usr/bin/env python
# coding=utf-8

import accelerate
import sys
import inspect
import argparse
import json
import logging
import math
import os
import numpy as np
from pathlib import Path

import datasets
import torch
import torch.nn as nn
from functools import partial
from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
)
from loguru import logger

from model.RetweetModel import (
    RetweetFeatureConfig,
    RetweetFeatureModel, 
    RetweetFusionModel, 
    RetweetFusionConfig
)
from train_utils import (
    feature_collator, 
    evaluation_loop, 
    label_scaling, 
    label_inverse_scaling,
    create_vocabulary_mapping,
    get_feature_dimensions,
    DENSE_FEATURES,
    SPARSE_FEATURES,
    VARLEN_SPARSE_FEATURES
)

def parse_args():
    parser = argparse.ArgumentParser(description="Train retweet prediction models")
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the feature dataset",
    )
    
    # Model arguments
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["feature_only", "fusion"],
        required=True,
        help="Type of model to train",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained Qwen3 model (required for fusion model)",
    )
    
    # Training arguments
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=256,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=256,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Initial learning rate to use.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay to use.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the final model."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="A seed for reproducible training."
    )
    
    # Model hyperparameters
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=32,
        help="Embedding dimension for sparse features"
    )
    parser.add_argument(
        "--dnn_hidden_units",
        type=str,
        default="2048,512,128",
        help="Hidden units for DNN layers (comma-separated)"
    )
    parser.add_argument(
        "--dnn_dropout",
        type=float,
        default=0.1,
        help="Dropout rate for DNN"
    )
    parser.add_argument(
        "--varlen_max_len",
        type=int,
        default=20,
        help="Maximum length for variable-length features"
    )
    parser.add_argument(
        "--label_log_scaling",
        action="store_true",
        help="Whether to use log scaling for labels"
    )
    
    # Logging arguments
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every X updates steps."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default="epoch",
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help="The integration to report the results and logs to.",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="retweet-prediction",
        help="The name of the wandb project.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="The name of the wandb run.",
    )
    
    args = parser.parse_args()
    
    # Parse hidden units
    args.dnn_hidden_units = [int(x) for x in args.dnn_hidden_units.split(',')]
    
    return args

def main():
    args = parse_args()
    
    # Initialize accelerator
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=True
            )
        ]
    )
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    
    # Set seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_from_disk(args.dataset_path)
    
    # Split dataset if needed
    train_dataset = dataset["train"]
    eval_dataset = dataset.get("eval", dataset.get("test", None))
    
    # Initialize tokenizer (for fusion model)
    tokenizer = None
    if args.model_type == "fusion":
        if not args.model_name_or_path:
            raise ValueError("--model_name_or_path is required for fusion model")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # Calculate feature dimensions from the actual data
    logger.info("Calculating feature dimensions...")
    dense_features_dim, sparse_feature_dims, varlen_feature_dims = get_feature_dimensions(train_dataset)

    logger.info(f"Dense features dimension: {dense_features_dim}")
    logger.info(f"Sparse features: {len(sparse_feature_dims)} features")
    logger.info(f"Variable-length features: {len(varlen_feature_dims)} features")

    # Get vocabulary mappings for variable-length features
    logger.info("Creating vocabulary mappings...")
    if not os.exists("./data/vocab_mappings.json"):
        vocab_mappings = create_vocabulary_mapping(train_dataset, VARLEN_SPARSE_FEATURES)

        # Save vocabulary mappings
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            with open("./data/vocab_mappings.json", "w") as f:
                json.dump(vocab_mappings, f)
    else:
        with open("./data/vocab_mappings.json", "r") as f:
            vocab_mappings = json.load(f)
    
    # Initialize model
    logger.info(f"Initializing {args.model_type} model...")
    
    if args.model_type == "feature_only":
        # Create configuration object
        config = RetweetFeatureConfig(
            dense_features_dim=len(DENSE_FEATURES),
            sparse_feature_dims=sparse_feature_dims,
            varlen_feature_dims=varlen_feature_dims,
            sparse_feature_names=SPARSE_FEATURES,
            varlen_feature_names=VARLEN_SPARSE_FEATURES,
            embedding_dim=args.embedding_dim,
            dnn_hidden_units=args.dnn_hidden_units,
            dnn_dropout=args.dnn_dropout,
            dnn_activation='relu',
            use_bn=True,
            l2_reg=1e-4,
            init_std=1e-4,
            varlen_pooling_modes=['mean', 'max']
        )
        
        # Initialize model with config
        model = RetweetFeatureModel(config)
        
        if accelerator.is_main_process:
            logger.info(f"Model arch: {model}")

    else:  # fusion
        config = RetweetFusionConfig.from_pretrained(
            args.model_name_or_path,
            dense_features_dim=dense_features_dim,
            sparse_features_dim=len(sparse_feature_dims),
            embedding_dim=args.embedding_dim,
            mlp_num=4,
            fusion_hidden_size=1024,
            dropout_rate=0.1
        )
        
        # Store the actual dimensions in the config for the model
        config.sparse_feature_dims = sparse_feature_dims
        config.varlen_feature_dims = varlen_feature_dims
        
        model = RetweetFusionModel.from_pretrained(
            args.model_name_or_path,
            config=config,
            ignore_mismatched_sizes=True
        )
    
    # Create data collator
    collate_fn = partial(
        feature_collator,
        tokenizer=tokenizer,
        use_text=(args.model_type == "fusion"),
        max_length=512,
        varlen_max_len=args.varlen_max_len
    )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    eval_dataloader = None
    if eval_dataset is not None:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
    
    # Label scaling
    scaler_info = None
    if args.label_log_scaling:
        logger.info("Applying label log scaling...")
        # Get all training labels
        train_labels = np.array(train_dataset['retweet_count'])
        scaler_info, scaled_labels = label_scaling(train_labels)
        # Update dataset with scaled labels
        train_dataset = train_dataset.add_column('scaled_labels', scaled_labels.tolist())
    
    # Optimizer
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    # Learning rate scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    
    # Prepare everything with accelerator
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    
    # Recalculate steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    # Initialize tracking
    if args.with_tracking:
        experiment_config = vars(args)
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        
        accelerator.init_trackers(
            project_name=args.project_name,
            config=experiment_config,
            init_kwargs={
                "wandb": {
                    "name": args.run_name,
                    "save_code": True,
                }
            }
        )
    
    # Training info
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Training loop
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        
        # Extract epoch from checkpoint path
        path = os.path.basename(checkpoint_path)
        if "epoch" in path:
            starting_epoch = int(path.split("_")[1]) + 1
            completed_steps = starting_epoch * num_update_steps_per_epoch
        progress_bar.update(completed_steps)
    
    # Training
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # Forward pass
                if args.model_type == "fusion":
                    outputs = model(
                        input_ids=batch.get('input_ids'),
                        attention_mask=batch.get('attention_mask'),
                        dense_features=batch['dense_features'],
                        sparse_features=batch['sparse_features'],
                        varlen_features=batch['varlen_features'],
                        labels=batch['labels']
                    )
                else:
                    outputs = model(
                        dense_features=batch['dense_features'],
                        sparse_features=batch['sparse_features'],
                        varlen_features=batch['varlen_features'],
                        labels=batch['labels']
                    )
                
                loss = outputs.loss
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                
                if completed_steps % args.logging_steps == 0:
                    avg_loss = accelerator.gather(total_loss).mean().item() / args.logging_steps
                    
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "train/loss": avg_loss,
                                "train/learning_rate": lr_scheduler.get_last_lr()[0],
                                "train/epoch": epoch,
                            },
                            step=completed_steps,
                        )
                    
                    logger.info(f"Step: {completed_steps}, Loss: {avg_loss:.4f}, LR: {lr_scheduler.get_last_lr()[0]:.2e}")
                    total_loss = 0
                
                if isinstance(args.checkpointing_steps, int) and completed_steps % args.checkpointing_steps == 0:
                    output_dir = f"checkpoint-{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
        
        # Evaluation
        if eval_dataloader is not None:
            logger.info(f"Running evaluation for epoch {epoch}...")
            metrics = evaluation_loop(
                model, 
                eval_dataloader, 
                accelerator,
                use_text=(args.model_type == "fusion")
            )
            
            logger.info(f"Epoch {epoch} evaluation metrics: {metrics}")
            
            if args.with_tracking:
                accelerator.log(
                    {f"eval/{k}": v for k, v in metrics.items()},
                    step=completed_steps,
                )
        
        # Save checkpoint at end of epoch
        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
    
    # Save final model
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        
        # Save model
        if accelerator.is_main_process:
            if hasattr(unwrapped_model, "save_pretrained"):
                unwrapped_model.save_pretrained(
                    args.output_dir,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                )
            else:
                torch.save(unwrapped_model.state_dict(), os.path.join(args.output_dir, "model.pt"))
            
            # Save tokenizer if used
            if tokenizer is not None:
                tokenizer.save_pretrained(args.output_dir)
            
            # Save scaler info if used
            if scaler_info is not None:
                with open(os.path.join(args.output_dir, "scaler_info.json"), "w") as f:
                    json.dump(scaler_info, f)
            
            # Save configuration
            with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
                json.dump(vars(args), f, indent=2, default=str)
    
    if args.with_tracking:
        accelerator.end_training()

if __name__ == "__main__":
    main()