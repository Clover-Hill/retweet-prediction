import sys
import inspect
import pdb 
import argparse
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

from functools import partial
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed

import datasets 
from datasets import load_dataset,load_from_disk
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from loguru import logger 

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

from model.KNNModel import KNNModel

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation of Retweet Prediction Model")
    parser.add_argument(
        "--dataset_name",
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
    
    # Train arguments
    
    parser.add_argument(
        "--project_name",
        type=str,
        default=None,
        help="The name of the project.",
    )
    parser.add_argument(
        "--group_name",
        type=str,
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
        "--do_eval",
        action="store_true",
    )
    parser.add_argument(
        "--do_test",
        action="store_true",
    )
    
    # KNN arguments
    
    parser.add_argument("--use_knn", action="store_true", help="whether to use knn method")
    parser.add_argument("--k", type=int, default=10, help="number of nearest neighbors to retrieve")
    parser.add_argument("--sim_threshold", type=float, default=0.7, help="IP similarity cut-off")
    parser.add_argument("--viral_threshold", type=float, default=0.20, help="ratio that triggers viral weighting")
    parser.add_argument("--index_name", type=str, default="feature_all", help="which FAISS index to use")
    parser.add_argument("--dim", type=int, default=2_048, help="dimension of feature_embedding")
    parser.add_argument("--use_gpu", action="store_true", help="search on GPU")
    
    args = parser.parse_args()
    return args
    
def main():
    args = parse_args()

    accelerator_log_kwargs = {}

    accelerator = Accelerator(**accelerator_log_kwargs)
    
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

    # -------------------------------------------------Load Model------------------------------------------------------------
    # Load config and model
    # if args.config_name:
    #     config = AutoConfig.from_pretrained(
    #         args.config_name,
    #         trust_remote_code=args.trust_remote_code,
    #     )
    # elif args.model_name_or_path:
    #     config = AutoConfig.from_pretrained(
    #         args.model_name_or_path,
    #         trust_remote_code=args.trust_remote_code,
    #     )
    # else:
    #     config = CONFIG_MAPPING[args.model_type]()
    #     logger.warning("You are instantiating a new config instance from scratch.")

    # model = AutoModelForCausalLM.from_pretrained( args.model_name_or_path, config=config)
    
    if args.use_knn:
        model = KNNModel(index_name=args.index_name, dim=args.dim, k=args.k, use_gpu=args.use_gpu, dataset_dir=args.dataset_name, sim_threshold=args.sim_threshold, viral_threshold=args.viral_threshold)
    
    # -------------------------------------------------Load Dataset------------------------------------------------------------
    feature_datasets = load_from_disk(args.dataset_name)
    
    # --------------------------------------------------Evaluation-----------------------------------------------------------
    
    if args.do_eval:
        eval_dataloader = DataLoader(
            feature_datasets["eval"], collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size, 
            shuffle=False,
            num_workers=1,
            prefetch_factor=4,
            pin_memory=True
        )
        # Create a partial function with your specific parameters
        model, eval_dataloader = accelerator.prepare( model, eval_dataloader )

        model.eval()
        loss = 0
        for batch in tqdm(eval_dataloader, desc = "Evaluating Model"):
            with torch.no_grad():
                outputs = model(feature_embedding=batch["feature_embedding"])
                
                # logger.info(f"retweet_count: {batch['retweet_count']}")
                # logger.info(f"pred_retweet_count: {outputs['pred_retweet_count']}")
                # logger.info(f"current loss: {loss.item() if isinstance(loss, torch.Tensor) else loss}")
                # pdb.set_trace()

                for i in range(args.per_device_eval_batch_size):
                    loss += torch.abs(batch["retweet_count"][i] - outputs["pred_retweet_count"][i])

        loss /= len(feature_datasets["eval"])
        if accelerator.is_main_process:
            logger.info(f"Evaluation Loss: {loss.item()}")
    
    # if args.do_test:
    #     eval_dataloader = DataLoader(
    #         lm_datasets, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size, 
    #         shuffle=False,
    #         num_workers=4,
    #         prefetch_factor=4,
    #         pin_memory=True
    #     )
    #     # Create a partial function with your specific parameters
    #     model, knn_generator ,eval_dataloader = accelerator.prepare( model, knn_generator, eval_dataloader )

    #     model.eval()
    #     knn_generator.eval()
    #     eval_lm = 0
    #     eval_joint = 0
    #     token_cnt = 0

    #     for batch in tqdm(eval_dataloader, desc = "Evaluating Model"):
    #         with torch.no_grad():
    #             outputs = model(
    #                 input_ids=batch["input_ids"],
    #                 attention_mask=batch["attention_mask"],
    #                 labels=None
    #             )
    #             knn_outputs = knn_generator(
    #                 input_ids=batch["input_ids"],
    #                 attention_mask=batch["attention_mask"],
    #                 labels=None
    #             )
                
    #             joint_loss, lm_loss, cnt = joint_evaluate(outputs.logits, knn_outputs.logits, batch, tokenizer, args)
    #             eval_joint += joint_loss.item()
    #             eval_lm += lm_loss.item()
    #             token_cnt += cnt
                
    #             # logger.info(f"lm_loss: {lm_loss.item()}")
    #             # logger.info(f"joint_loss: {joint_loss.item()}")
    #             # logger.info(f"cnt : {cnt}")

    #     eval_lm = torch.tensor(eval_lm).to(model.device)
    #     eval_joint = torch.tensor(eval_joint).to(model.device)
    #     token_cnt = torch.tensor(token_cnt).to(model.device)
        
    #     eval_lm = accelerator.gather(eval_lm).sum().item()
    #     eval_joint = accelerator.gather(eval_joint).sum().item()
    #     token_cnt = accelerator.gather(token_cnt).sum().item()

    #     eval_lm = math.exp(eval_lm / token_cnt)
    #     eval_joint = math.exp(eval_joint / token_cnt)
    #     logger.info(f"token count: {token_cnt}")
        
    #     if accelerator.is_main_process:
    #         print(f"lm perplexity: {eval_lm}")
    #         print(f"joint perplexity: {eval_joint}")

if __name__ == "__main__":
    main()