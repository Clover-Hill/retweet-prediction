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
import csv
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

from model.RetweetModel import RetweetConfig, RetweetBaseModel, RetweetClassificationModel, RetweetRegressionModel, RetweetMultiRegressionModel
from train_utils import feature_collator, evaluation_loop

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--classification_model_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--regression_model_path",
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
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
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
    
    # Qwen3 Args
    
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
        "--head_type",
        type=str,
        default=None,
        help="type of language model head, regression or classification",
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
    parser.add_argument(
        "--do_test",
        action="store_true",
    )
    parser.add_argument(
        "--mlp_num",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--scalar_features_dim",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--viral_threshold",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--viral_mean_val",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--neg_pos_ratio",
        type=float,
        default=None,
    )
    
    args = parser.parse_args()

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
    # Load config and model
    classification_model = RetweetClassificationModel.from_pretrained( args.classification_model_path )
    
    regression_model = RetweetMultiRegressionModel.from_pretrained( args.regression_model_path )
    
    # -------------------------------------------------Preprocess Dataset------------------------------------------------------------
    # Default to use gpt tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.classification_model_path)
    retweet_datasets = load_from_disk(args.dataset_name)
    
    # --------------------------------------------------Evaluation-----------------------------------------------------------
    classification_collate_fn = partial(
        feature_collator,
        tokenizer=tokenizer,
        use_rich_text=False
    )
    
    regression_collate_fn = partial(
        feature_collator,
        tokenizer=tokenizer,
        use_rich_text=True
    )

    with open("/fs-computility/plm/shared/jqcao/projects/retweet-prediction/feature_engineering/count_intervals.json", 'r') as f:
        intervals = json.load(f)
    weight = torch.tensor([interval['mean_val'] for interval in intervals], dtype=torch.float32)
    
    if args.do_eval:
        classification_dataloader = DataLoader(
            retweet_datasets["eval"], collate_fn=classification_collate_fn, batch_size=args.per_device_eval_batch_size, 
            shuffle=False,
            num_workers=4,
            prefetch_factor=4,
            pin_memory=True
        )
        regression_dataloader = DataLoader(
            retweet_datasets["eval"], collate_fn=regression_collate_fn, batch_size=args.per_device_eval_batch_size, 
            shuffle=False,
            num_workers=4,
            prefetch_factor=4,
            pin_memory=True
        )
        # Create a partial function with your specific parameters
        classification_model, regression_model, classification_dataloader, regression_dataloader = accelerator.prepare( classification_model, regression_model, classification_dataloader, regression_dataloader )

        all_regression_logits = []
        all_classification_logits = []
        all_retweet_counts = []

        classification_model.eval()
        regression_model.eval()
        
        for batch in tqdm(classification_dataloader, desc = "Evaluating Classification Model", disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                classification_outputs = classification_model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    scalar_features=batch['scalar_features'],
                )
                classification_logits = classification_outputs.logits
                classification_logits = torch.sigmoid(classification_logits)
                
                # Gather predictions and labels from all processes
                gathered_logits = accelerator.gather_for_metrics(classification_logits)
                gathered_retweet_counts = accelerator.gather_for_metrics(batch['retweet_count'])
                
                all_classification_logits.extend(gathered_logits.cpu().numpy().tolist())
                all_retweet_counts.extend(gathered_retweet_counts.cpu().numpy().tolist())
        
        for batch in tqdm(regression_dataloader, desc = "Evaluating Regression Model", disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                regression_outputs = regression_model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                )
                regression_logits = regression_outputs.logits
                regression_logits = torch.softmax(regression_logits, dim=-1)
                
                # Gather predictions and labels from all processes
                gathered_logits = accelerator.gather_for_metrics(regression_logits)
                
                all_regression_logits.extend(gathered_logits.cpu().numpy().tolist())

        # Only compute metrics on main process
        if accelerator.is_main_process:
            all_classification_logits = torch.tensor(all_classification_logits)
            all_retweet_counts = torch.tensor(all_retweet_counts)
            
            classification_predictions = (all_classification_logits > args.viral_threshold).float()
            mae_loss = 0
            
            for i in range(len(all_retweet_counts)):
                if classification_predictions[i] == 1:
                    prediction_class = torch.argmax(torch.tensor(all_regression_logits[i]), dim=-1)

                    prediction = torch.sum(torch.tensor(all_regression_logits[i]) * weight, dim=-1)
                    # prediction = intervals[prediction_class]["mean_val"]
                    
                    print(f"Probability: {all_regression_logits[i]}")
                    print(f"Prediction: {prediction}, Actual: {all_retweet_counts[i]}")

                    mae_loss += abs(all_retweet_counts[i] - prediction)
                else:
                    mae_loss += abs(all_retweet_counts[i] - 0)
            
            logger.info(f"MAE Loss: {mae_loss / len(all_retweet_counts)}")
            
        accelerator.wait_for_everyone()
        return

    if args.do_test:
        classification_dataloader = DataLoader(
            retweet_datasets["test"], collate_fn=classification_collate_fn, batch_size=args.per_device_eval_batch_size, 
            shuffle=False,
            num_workers=4,
            prefetch_factor=4,
            pin_memory=True
        )
        regression_dataloader = DataLoader(
            retweet_datasets["test"], collate_fn=regression_collate_fn, batch_size=args.per_device_eval_batch_size, 
            shuffle=False,
            num_workers=4,
            prefetch_factor=4,
            pin_memory=True
        )
        # Create a partial function with your specific parameters
        classification_model, regression_model, classification_dataloader, regression_dataloader = accelerator.prepare( classification_model, regression_model, classification_dataloader, regression_dataloader )

        all_regression_logits = []
        all_classification_logits = []
        classification_model.eval()
        regression_model.eval()
        
        for batch in tqdm(classification_dataloader, desc = "Evaluating Classification Model", disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                classification_outputs = classification_model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    scalar_features=batch['scalar_features'],
                )
                classification_logits = classification_outputs.logits
                classification_logits = torch.sigmoid(classification_logits)
                
                # Gather predictions and labels from all processes
                gathered_logits = accelerator.gather_for_metrics(classification_logits)
                
                all_classification_logits.extend(gathered_logits.cpu().numpy().tolist())
        
        for batch in tqdm(regression_dataloader, desc = "Evaluating Regression Model", disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                regression_outputs = regression_model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                )
                regression_logits = regression_outputs.logits
                regression_logits = torch.softmax(regression_logits, dim=-1)
                
                # Gather predictions and labels from all processes
                gathered_logits = accelerator.gather_for_metrics(regression_logits)
                
                all_regression_logits.extend(gathered_logits.cpu().numpy().tolist())

        # Only compute metrics on main process
        if accelerator.is_main_process:
            all_classification_logits = torch.tensor(all_classification_logits)
            
            classification_predictions = (all_classification_logits > args.viral_threshold).float()
            
            all_tweet_ids = []
            predicted_retweet_counts = []

            for i in range(len(retweet_datasets["test"])):
                if classification_predictions[i] == 1:
                    prediction_class = torch.argmax(torch.tensor(all_regression_logits[i]), dim=-1)
                    prediction = torch.sum(torch.tensor(all_regression_logits[i]) * weight, dim=-1)
                    
                    predicted_retweet_counts.append(prediction.item())
                else:
                    predicted_retweet_counts.append(0)
                
                all_tweet_ids.append(retweet_datasets["test"][i]["id"])
            
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            
            # Write results to txt file
            output_file = os.path.join(args.output_dir, 'test_predictions.csv')
            with open(output_file, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(["TweetID", "NoRetweets"])
                for tweet_id, prediction in zip(all_tweet_ids, predicted_retweet_counts):
                    writer.writerow([str(int(tweet_id)), str(prediction)])
            
            logger.info(f"Saved test predictions to {output_file}")
            
        accelerator.wait_for_everyone()
        return

if __name__ == "__main__":
    main()