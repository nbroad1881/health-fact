#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
import argparse
import math

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import precision_recall_fscore_support, f1_score

import datasets
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
    get_scheduler,
)
from transformers.integrations import WandbCallback
from transformers.trainer_utils import get_last_checkpoint
from accelerate import Accelerator

from data import DataModule
from utils import set_wandb_env_vars, get_configs, NewWandbCB, reinit_model_weights
from bertlstm import BERTLSTM

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)

    return parser

def main():

    parser = get_parser()
    args = parser.parse_args()

    cfg, train_args = get_configs(args.config_file)
    set_wandb_env_vars(cfg)

    training_args = TrainingArguments(**train_args)
    
    accelerator = (
        Accelerator(logging_dir=training_args.output_dir, mixed_precision="fp16")
    )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    data_module = DataModule(cfg)

    data_module.prepare_dataset()

    tokenized_ds = data_module.tokenized_dataset

    id2labels = {
        0: "false",
        1: "mixture",
        2: "true",
        3: "unproven",
    }

    config = AutoConfig.from_pretrained(
        cfg["model_name_or_path"],
        num_labels=len(id2labels),
        finetuning_task="text-classification",
    )
    
    config.update({
        "block_size": cfg.get("block_size", 64),
        "num_random_blocks": cfg.get("num_random_blocks", 3),
        "lstm": True
    })

    model = BERTLSTM(config)
    
    model.transformer = AutoModel.from_pretrained(cfg["model_name_or_path"])

    reinit_model_weights(model, cfg.get("reinit_layers", 0), config)

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(tokenized_ds["train"])), 3):
            logger.info(
                f"Sample {index} of the training set: {tokenized_ds['train'][index]}."
            )

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=-1)

        _, _, f1s, _ = precision_recall_fscore_support(p.label_ids, preds)
        micro_f1 = f1_score(p.label_ids, preds, average="micro")
        macro_f1 = f1_score(p.label_ids, preds, average="macro")

        metrics = {
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            **{f"accuracy_{id2labels[i]}": f1s[i] for i in range(len(f1s))},
        }

        return metrics

    data_collator = DataCollatorWithPadding(data_module.tokenizer, pad_to_multiple_of=cfg["pad_multiple"])

    train_dataloader = DataLoader(
        tokenized_ds["train"],
        batch_size=training_args.per_device_train_batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=data_collator,
    )
    
    eval_dataloader = DataLoader(
        tokenized_ds["validation"],
        batch_size=training_args.per_device_eval_batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=data_collator,
    )
        
         # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    if training_args.max_steps is None:
        training_args.max_steps = training_args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_steps = True

    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.max_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )


    # # Figure out how many steps we should save the Accelerator states
    # if hasattr(args.checkpointing_steps, "isdigit"):
    #     checkpointing_steps = args.checkpointing_steps
    #     if args.checkpointing_steps.isdigit():
    #         checkpointing_steps = int(args.checkpointing_steps)
    # else:
    #     checkpointing_steps = None

    # We need to initialize the trackers we use, and also store our configuration.
    # We initialize the trackers only on main process because `accelerator.log`
    # only logs on main process and we don't want empty logs/runs on other processes.
    # if "wandb" in training_args.report_to:
    #     if accelerator.is_main_process:
    #         experiment_config = vars(args)
    #         # TensorBoard cannot log Enums, need the raw value
    #         experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
    #         accelerator.init_trackers("glue_no_trainer", experiment_config)

    # Train!
    total_batch_size = training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(tokenized_ds['train'])}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {training_args.max_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(training_args.max_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    for epoch in range(starting_epoch, training_args.num_train_epochs):
        model.train()

        for step, batch in enumerate(train_dataloader):
            # # We need to skip steps until we reach the resumed step
            # if args.resume_from_checkpoint and epoch == starting_epoch:
            #     if resume_step is not None and step < resume_step:
            #         completed_steps += 1
            #         continue
            outputs = model(
                input_ids=batch["input_ids"], 
                attention_mask=batch["attention_mask"], 
                doc_ids=batch["idx"],
                labels=batch["labels"],
            )
            loss = outputs.loss

            loss = loss / training_args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % training_args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
            
            if step % 50 == 0:
                print(loss)

            # if isinstance(checkpointing_steps, int):
            #     if completed_steps % checkpointing_steps == 0:
            #         output_dir = f"step_{completed_steps }"
            #         if training_args.output_dir is not None:
            #             output_dir = os.path.join(training_args.output_dir, output_dir)
            #         accelerator.save_state(output_dir)

            if completed_steps >= training_args.max_steps:
                break

        model.eval()
        samples_seen = 0
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(
                input_ids=batch["input_ids"], 
                attention_mask=batch["attention_mask"], 
                doc_ids=batch["idx"],
                labels=batch["labels"],
            )
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather((predictions, batch["labels"]))
            # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                    references = references[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]
        #     metric.add_batch(
        #         predictions=predictions,
        #         references=references,
        #     )

        # eval_metric = metric.compute()
        # logger.info(f"epoch {epoch}: {eval_metric}")

        # if args.with_tracking:
        #     accelerator.log(
        #         {
        #             "accuracy" if args.task_name is not None else "glue": eval_metric,
        #             "train_loss": total_loss.item() / len(train_dataloader),
        #             "epoch": epoch,
        #             "step": completed_steps,
        #         },
        #         step=completed_steps,
        #     )


    if training_args.do_predict:
        logger.info("*** Predict ***")




if __name__ == "__main__":
    main()
