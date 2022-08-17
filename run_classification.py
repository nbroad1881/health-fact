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
""" Finetuning the library models for sequence classification"""

import logging
import os
import random
import sys

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, f1_score

import hydra
from omegaconf import DictConfig
import datasets
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.integrations import WandbCallback
from transformers.trainer_utils import get_last_checkpoint

from data import DataModule
from utils import WandbCallbackV2, set_wandb_env_vars, set_mlflow_env_vars


logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    train_args = dict(cfg.training_arguments)
    del cfg.training_arguments
    

    training_args = TrainingArguments(**train_args)

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

    # If running distributed, this will do something on the main process, while
    #    blocking replicas, and when it's finished, it will release the replicas.
    with training_args.main_process_first(desc="Dataset loading and tokenization"):
        data_module.prepare_dataset()

    id2label = data_module.id2label
    label2id = data_module.label2id

    config = AutoConfig.from_pretrained(
        cfg.model.model_name_or_path,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        finetuning_task="text-classification",
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.model_name_or_path,
        config=config,
        revision=cfg.model.model_revision,
        use_auth_token=cfg.model.use_auth_token,
    )

    train_dataset = data_module.get_train_dataset()
    eval_dataset = data_module.get_eval_dataset() if training_args.do_eval else None

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(
                f"Sample {index} of the training set: {train_dataset[index]}."
            )

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=-1)

        _, _, f1s, _ = precision_recall_fscore_support(p.label_ids, preds)
        micro_f1 = f1_score(p.label_ids, preds, average="micro")
        macro_f1 = f1_score(p.label_ids, preds, average="macro")

        # f1 and accuracy are the same when doing micro-f1
        # accuracy is more recognizable
        accuracies = {f"{id2label[i]}_accuracy": f1s[i] for i in range(len(f1s))}
        accuracies["macro_accuracy"] = sum(accuracies.values())/len(accuracies.values())

        metrics = {
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            **accuracies,
        }

        return metrics

    data_collator = DataCollatorWithPadding(
        data_module.tokenizer, pad_to_multiple_of=cfg.data.pad_multiple
    )

    callbacks = []
    if "wandb" in training_args.report_to:
        callbacks.append(WandbCallbackV2(dict(cfg)))

    # mlflow callback will automatically be applied

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=data_module.tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    trainer.remove_callback(WandbCallback)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_predict:
        test_ds = data_module.get_test_dataset()
        logger.info("*** Predict ***")

        metrics = trainer.predict(test_ds).metrics
        metrics["eval_samples"] = len(test_ds)

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

    # Will create nice README.md file even if not pushing to hub
    kwargs = {
        "finetuned_from": cfg.model.model_name_or_path,
        "tasks": cfg.task,
        "language": cfg.language,
        "dataset_tags": cfg.data.dataset_name,
    }

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
