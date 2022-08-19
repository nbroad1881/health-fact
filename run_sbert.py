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
""" Finetuning the sbert chunks for long sequence classification"""

import logging
import os
import math
import sys
import logging
from itertools import chain
from functools import partial

import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import f1_score

import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig
import datasets
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
    get_scheduler,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_pt_utils import IterableDatasetShard
from accelerate import Accelerator

from utils import set_wandb_env_vars, set_mlflow_env_vars
from sbert import SBertLongformer, SBertCollate


logger = logging.getLogger(__name__)


def tokenize(example, tokenizer, max_length, stride):

    tokenized_example = tokenizer(
        example["claim"],
        example["main_text"],
        padding=True,
        truncation=True,
        max_length=max_length,
        doc_stride=stride,
        return_overflowing_tokens=True,
    )

    tokenized_example["label"] = example["label"]
    tokenized_example["length"] = len(tokenized_example["input_ids"])

    return tokenized_example


def training_loop(
    accelerator, model, optimizer, lr_scheduler, train_dataloader, eval_dataloader, args
):

    progress_bar = tqdm(
        range(args.max_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0

    for epoch in range(args.num_train_epochs):

        model.train()

        if args.report_to is not None:
            total_loss = 0

        for step, batch in enumerate(train_dataloader):

            outputs = model(**batch)
            loss = outputs.loss

            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()

            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if args.save_strategy == "steps":
                checkpointing_steps = args.save_steps
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_steps:
                break

        metrics = eval_loop(
            model=model,
            eval_dataloader=eval_dataloader,
        )

        metrics.update({
            "epoch": epoch,
            "step": completed_steps
        })

        accelerator.log(metrics, step=completed_steps)



def eval_loop(model, eval_dataloader):

    model.eval()

    y_preds = []
    y_true = []
    eval_loss = 0
    for batch in eval_dataloader:

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            lengths=batch["lengths"],
        )

        eval_loss += outputs["loss"].detach().float()
        logits = outputs["logits"]
        preds = logits.argmax(-1).detach().cpu().tolist()
        labels = batch["labels"].detach().cpu().tolist()

        y_preds.append(preds)
        y_true.append(labels)

    y_preds = list(chain(*y_preds))
    y_true = list(chain(*labels))

    f1_micro = f1_score(y_true, y_preds, average="micro")
    f1_macro = f1_score(y_true, y_preds, average="macro")

    return {
        "eval_loss": eval_loss,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
    }


def num_examples(dataloader: DataLoader) -> int:
    """
    Helper to get number of samples in a [`~torch.utils.data.DataLoader`] by accessing its dataset. When
    dataloader.dataset does not exist or has no length, estimates as best it can
    """
    dataset = dataloader.dataset
    # Special case for IterableDatasetShard, we need to dig deeper
    if isinstance(dataset, IterableDatasetShard):
        return len(dataloader.dataset.dataset)
    return len(dataloader.dataset)


def train_samples_steps_epochs(dataloader, args):
    total_train_batch_size = (
        args.train_batch_size * args.gradient_accumulation_steps * args.world_size
    )

    len_dataloader = len(dataloader)
    num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    num_examples = num_examples(dataloader)
    if args.max_steps > 0:
        max_steps = args.max_steps
        num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
            args.max_steps % num_update_steps_per_epoch > 0
        )
        # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
        # the best we can do.
        num_train_samples = args.max_steps * total_train_batch_size
    else:
        max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
        num_train_epochs = math.ceil(args.num_train_epochs)
        num_train_samples = num_examples(dataloader) * args.num_train_epochs

    return num_train_samples, max_steps, num_train_epochs


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    train_args = dict(cfg.training_arguments)
    del cfg.training_arguments

    training_args = TrainingArguments(**train_args)

    mixed_precision = "no"
    if training_args.fp16:
        mixed_precision = "fp16"
    if training_args.bf16:
        mixed_precision = "bf16"

    accelerator = Accelerator(
        mixed_precision=mixed_precision, log_with=training_args.report_to
    )

    if "wandb" in training_args.report_to:
        set_wandb_env_vars(cfg)
    if "mlflow" in training_args.report_to:
        set_mlflow_env_vars(cfg)

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

    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_name_or_path,
    )

    raw_dataset = datasets.load_dataset(
        cfg.data.dataset_name, cfg.data.dataset_config_name
    )
    raw_dataset = raw_dataset.filter(lambda x: x["label"] != -1)

    if cfg.data.n_rows > 0:
        for split in raw_dataset:
            raw_dataset[split] = raw_dataset[split].select(range(cfg.data.n_rows))

    cols = raw_dataset["train"].column_names

    # If running distributed, this will do something on the main process, while
    #    blocking replicas, and when it's finished, it will release the replicas.
    with training_args.main_process_first(desc="Dataset loading and tokenization"):
        tokenized_dataset = raw_dataset.map(
            partial(tokenize, tokenizer=tokenizer),
            num_proc=cfg.num_proc,
            batched=False,
            remove_columns=cols,
        )

    collator = SBertCollate(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        tokenized_dataset["train"],
        batch_size=training_args.per_device_train_batch_size,
        collate_fn=collator,
        drop_last=training_args.dataloader_drop_last,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=training_args.dataloader_pin_memory,
    )

    eval_dataloader = DataLoader(
        tokenized_dataset["validation"],
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=collator,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=training_args.dataloader_pin_memory,
    )

    # Start with the config for the sbert model
    model_config = AutoConfig.from_pretrained(cfg.model.model_name_or_path)
    model_config.update(
        {
            "sbert_model": cfg.model.model_name_or_path,
            "num_labels": len(raw_dataset["train"].unique("label")),
            "hidden_act": cfg.model.hidden_act,
            "intermediate_size": cfg.model.intermediate_size,
            "num_attention_heads": cfg.model.num_attention_heads,
            "num_hidden_layers": cfg.model.num_hidden_layers,
        }
    )

    model = SBertLongformer.from_pretrained(
        cfg.model.model_name_or_path, config=model_config
    )

    optim_cls, optim_args = Trainer.get_optimizer_cls_and_kwargs(training_args)

    optimizer = optim_cls(model.parameters(), **optim_args)

    num_train_samples, max_steps, num_train_epochs = train_samples_steps_epochs(
        train_dataloader, training_args
    )

    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.get_warmup_steps(max_steps),
        num_training_steps=max_steps,
    )

    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # We initialize the trackers only on main process because `accelerator.log`
    # only logs on main process and we don't want empty logs/runs on other processes.
    if training_args.report_to:
        if accelerator.is_main_process:
            experiment_config = vars(cfg)
            # TensorBoard cannot log Enums, need the raw value
            experiment_config["lr_scheduler_type"] = experiment_config[
                "lr_scheduler_type"
            ].value
            accelerator.init_trackers(cfg.project_name, experiment_config)

    training_loop(
        accelerator, model, optimizer, lr_scheduler, train_dataloader, eval_dataloader, training_args
    )

    accelerator.end_training()


if __name__ == "__main__":
    main()
