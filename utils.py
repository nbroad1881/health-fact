import os
import re
import yaml
from pathlib import Path

import torch
from transformers.integrations import WandbCallback
from transformers.utils import logging
from transformers.file_utils import is_torch_tpu_available

logger = logging.get_logger(__name__)

def set_wandb_env_vars(cfg):
    """
    Set environment variables from the config dict object.
    The environment variables can be picked up by wandb in Trainer.
    """

    os.environ["WANDB_ENTITY"] = getattr(cfg.wandb, "entity", "")
    os.environ["WANDB_PROJECT"] = getattr(cfg.wandb, "project", "")
    os.environ["WANDB_RUN_GROUP"] = getattr(cfg.wandb, "group", "")
    os.environ["WANDB_JOB_TYPE"] = getattr(cfg.wandb, "job_type", "")
    os.environ["WANDB_NOTES"] = getattr(cfg.wandb, "notes", "")
    if cfg.wandb.tags:
        os.environ["WANDB_TAGS"] = ",".join(cfg.wandb.tags)
        
def set_mlflow_env_vars(cfg):
    """
    Set environment variables from the config object.
    The environment variables can be picked up by mlflow in Trainer.
    """
    
    os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = cfg.mlflow.log_artifact
    os.environ["MLFLOW_NESTED_RUN"] = cfg.mlflow.nested_run
    os.environ["MLFLOW_EXPERIMENT_NAME"] = cfg.mlflow.experiment_name
    os.environ["MLFLOW_FLATTEN_PARAMS"] = cfg.mlflow.flatten_params
    os.environ["MLFLOW_RUN_ID"] = cfg.mlflow.run_id


class WandbCallbackV2(WandbCallback):
    """
    The current WandbCallback doesn't read some environment variables.
    This implementation fixes that.

    https://github.com/huggingface/transformers/blob/d6eeb871706db0d64ab9ffd79f9545d95286b536/src/transformers/integrations.py#L566-L641
    """
    def __init__(self, run_config):
        super().__init__()
        self.run_config = run_config

    def setup(self, args, state, model, **kwargs):
        """
        Setup the optional Weights & Biases (*wandb*) integration.
        One can subclass and override this method to customize the setup if needed. Find more information
        [here](https://docs.wandb.ai/integrations/huggingface). You can also override the following environment
        variables:
        Environment:
            WANDB_LOG_MODEL (`bool`, *optional*, defaults to `False`):
                Whether or not to log model as artifact at the end of training. Use along with
                *TrainingArguments.load_best_model_at_end* to upload best model.
            WANDB_WATCH (`str`, *optional* defaults to `"gradients"`):
                Can be `"gradients"`, `"all"` or `"false"`. Set to `"false"` to disable gradient logging or `"all"` to
                log gradients and parameters.
            WANDB_PROJECT (`str`, *optional*, defaults to `"huggingface"`):
                Set this to a custom string to store results in a different project.
            WANDB_DISABLED (`bool`, *optional*, defaults to `False`):
                Whether or not to disable wandb entirely. Set *WANDB_DISABLED=true* to disable.
        """
        if self._wandb is None:
            return
        self._initialized = True
        if state.is_world_process_zero:
            logger.info(
                'Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"'
            )
            combined_dict = {**args.to_sanitized_dict(), **self.run_config}

            if hasattr(model, "config") and model.config is not None:
                model_config = model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}
            run_name = os.getenv("WANDB_NAME")

            if self._wandb.run is None:
                tags = os.getenv("WANDB_TAGS", None)
                save_code = os.getenv("WANDB_DISABLE_CODE", None)

                # environment variables get priority
                self._wandb.init(
                    project=os.getenv("WANDB_PROJECT", "huggingface"),
                    name=run_name,
                    group=os.getenv("WANDB_RUN_GROUP"),
                    notes=os.getenv("WANDB_NOTES", None),
                    entity=os.getenv("WANDB_ENTITY", None),
                    id=os.getenv("WANDB_RUN_ID", None),
                    dir=os.getenv("WANDB_DIR", None),
                    tags=tags.split(",") if tags else None,
                    job_type=os.getenv("WANDB_JOB_TYPE", None),
                    mode=os.getenv("WANDB_MODE", None),
                    anonymous=os.getenv("WANDB_ANONYMOUS", None),
                    save_code=bool(save_code) if save_code is not None else save_code,
                )
            # add config parameters (run may have been created manually)
            self._wandb.config.update(combined_dict, allow_val_change=True)

            # define default x-axis (for latest wandb versions)
            if getattr(self._wandb, "define_metric", None):
                self._wandb.define_metric("train/global_step")
                self._wandb.define_metric(
                    "*", step_metric="train/global_step", step_sync=True
                )

            # # keep track of model topology and gradients, unsupported on TPU
            if not is_torch_tpu_available() and os.getenv("WANDB_WATCH") != "false":
                self._wandb.watch(
                    model,
                    log=os.getenv("WANDB_WATCH", "gradients"),
                    log_freq=max(100, args.logging_steps),
                )


def reinit_model_weights(model, n_layers, config):
    """
    Revisting few-sample BERT tuning showed that re-initializing
    the last few layers can be beneficial. This function helps do that.

    https://github.com/asappresearch/revisit-bert-finetuning
    """

    backbones = {
        "big_bird": "bert",
        "deberta-v2": "deberta",
        "bart": "model",
    }
    
    if config.lstm:
        backbone_name = "transformer"
    else:
        backbone_name = backbones.get(config.model_type, config.model_type)

    backbone = getattr(model, backbone_name)
    if config.model_type == "bart":
        std = config.init_std
    else:
        std = config.initializer_range

    if n_layers > 0:
        if config.model_type == "bart":
            decoder_layers = backbone.decoder.layers
            reinit_layers(decoder_layers, n_layers, std)
        else:
            encoder_layers = backbone.encoder.layer
            reinit_layers(encoder_layers, n_layers, std)

    if config.model_type == "bart":
        output = [model.classification_head]
    else:
        output = [model.classifier]
        if config.lstm:
            output.append(model.bilstm)

    reinit_modules(output, std)


def reinit_layers(layers, n_layers, std):
    """
    Helper function to reinitialize last n_layers
    """
    for layer in layers[-n_layers:]:
        reinit_modules(layer.modules(), std)


def reinit_modules(modules, std, reinit_embeddings=False):
    """
    Reinitializes every Linear, Embedding, and LayerNorm module provided.
    """
    for module in modules:
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif reinit_embeddings and isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)