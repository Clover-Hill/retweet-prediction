#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import accelerate
import sys
import inspect
import pdb 
import argparse
import json
import logging
import math
import time
import os
import random
from itertools import chain
import numpy as np
from pathlib import Path
import pickle
import torch.nn as nn 
from datasets import Dataset
from transformers.pytorch_utils import Conv1D

import datasets
import torch
from functools import partial
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset,load_from_disk
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
import torch.nn.functional as F
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

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
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
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "Whether to trust the execution of code from datasets/models defined on the Hub."
            " This option should only be set to `True` for repositories you trust and in which you have read the"
            " code, as it will execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
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
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    
    parser.add_argument(
        "--project_name",
        type=str,
        default=None,
        help="The name of the project.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="The name of the run.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        help="Logging steps",
    )
    parser.add_argument(
        "--from_scratch",
        action="store_true",
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
    )
    
    # Model hyperparameters
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["feature_only", "fusion"],
        required=True,
        help="Type of model to train",
    )

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
    
    args = parser.parse_args()

    # Parse hidden units
    args.dnn_hidden_units = [int(x) for x in args.dnn_hidden_units.split(',')]

    return args

def main():
    args = parse_args()

    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs, kwargs_handlers=[
        accelerate.DistributedDataParallelKwargs(
            find_unused_parameters=True
        )
    ])
    
    if args.report_to == "wandb":
        accelerator.init_trackers(
            project_name=args.project_name, 
            config=args,
            init_kwargs={
                "wandb": {
                    "name": args.run_name if args.run_name is not None else None,
                    "save_code": True,
                },
            }
        )

    # Make one log on every process with the configuration for debugging.
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
        log_level = logging.INFO
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        log_level = logging.ERROR

    logger.remove()
    logger.add(sys.stdout, format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <blue>{process.name}</blue> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level=log_level)

    # Intercept default logging and transform to loguru
    class InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            # Get corresponding Loguru level if it exists.
            level: str | int
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message.
            frame, depth = inspect.currentframe(), 0
            while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    logging.basicConfig(handlers=[InterceptHandler()], level=log_level, force=True)
    transformers.utils.logging.disable_default_handler()
    transformers.utils.logging.add_handler(InterceptHandler())
    logger.info(f"{accelerator.state}")

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # -------------------------------------------------Load Dataset and Model------------------------------------------------------------
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_from_disk(args.dataset_path)
    
    # Split dataset if needed
    train_dataset = dataset["train"].select(range(1000))
    eval_dataset = train_dataset
    # eval_dataset = dataset.get("eval", dataset.get("test", None))
    
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
    if not os.path.exists("./data/vocab_mappings.json"):
        vocab_mappings = create_vocabulary_mapping(train_dataset, VARLEN_SPARSE_FEATURES)

        # Save vocabulary mappings
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            with open("./data/vocab_mappings.json", "w") as f:
                json.dump(vocab_mappings, f, indent=4, ensure_ascii=False)
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
            varlen_pooling_modes=['mean']
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
    
    # -------------------------------------------------Preprocess Dataset------------------------------------------------------------
    # Default to use gpt tokenizer
    # Initialize tokenizer (for fusion model)
    tokenizer = None
    if args.model_type == "fusion":
        if not args.model_name_or_path:
            raise ValueError("--model_name_or_path is required for fusion model")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        
        
    # Create data collator
    collate_fn = partial(
        feature_collator,
        tokenizer=tokenizer,
        use_text=(args.model_type == "fusion"),
        max_length=512,
        varlen_max_len=args.varlen_max_len,
        vocab_mappings = vocab_mappings,
    )
    # --------------------------------------------------Start Training-----------------------------------------------------------
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

    len_train_dataset = len(train_dataset)
    
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
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

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader)  / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6}M")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader) 
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader) 

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    logging_interval_loss = 0
    total_loss = 0

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):
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
                logging_interval_loss += loss.detach().float()

                accelerator.backward(loss)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # Clip gradnorm
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                completed_steps += 1
                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    actual_steps = args.gradient_accumulation_steps if not accelerator.gradient_state.end_of_dataloader else len(active_dataloader) % args.gradient_accumulation_steps

                    if actual_steps == 0:
                        actual_steps = args.gradient_accumulation_steps

                    avg_loss = accelerator.gather(logging_interval_loss).mean().item() / actual_steps / args.logging_steps
                    total_loss += accelerator.gather(logging_interval_loss).mean().item() / actual_steps 

                    to_be_logged = {
                        "train/learning_rate": lr_scheduler.get_last_lr()[0],
                        "train/train_loss": avg_loss,
                        "train/grad_norm": grad_norm,
                        "train/epoch": epoch,
                        "train/rolling_loss":total_loss / completed_steps,
                    }
                    accelerator.log(to_be_logged,step=completed_steps)
                    if accelerator.is_main_process:
                        logger.info(f"step: {completed_steps}, loss: {avg_loss}, lr: {lr_scheduler.get_last_lr()[0]}")

                    logging_interval_loss = 0

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)

                    # Save state for continue training
                    accelerator.save_state(output_dir)

                    # Save unwrap model
                    # unwrapped_model = accelerator.unwrap_model(model)
                    # unwrapped_model.save_pretrained(
                    #       output_dir,
                    #       is_main_process=accelerator.is_main_process,
                    #       save_function=accelerator.save,
                    #       state_dict=accelerator.get_state_dict(model)
                    # )

                    # if accelerator.is_main_process:
                    #     unwrapped_model.config.save_pretrained(output_dir)
                    #     tokenizer.save_pretrained(output_dir)

            if completed_steps >= args.max_train_steps:
                break
        
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
            
        # if args.checkpointing_steps == "epoch":
        #     output_dir = f"epoch_{epoch}"
        #     if args.output_dir is not None:
        #         output_dir = os.path.join(args.output_dir, output_dir)

        #     # Save state for continue training
        #     accelerator.save_state(output_dir)

            # Save unwrap model
            # unwrapped_model = accelerator.unwrap_model(model)
            # unwrapped_model.save_pretrained(
            #       output_dir,
            #       is_main_process=accelerator.is_main_process,
            #       save_function=accelerator.save,
            #       state_dict=accelerator.get_state_dict(model)
            # )

            # if accelerator.is_main_process:
            #     unwrapped_model.config.save_pretrained(output_dir)
            #     tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()