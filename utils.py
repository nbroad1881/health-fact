import os
import re
import yaml
from pathlib import Path

from transformers.integrations import WandbCallback
from transformers.utils import logging
from transformers.file_utils import is_torch_tpu_available

logger = logging.get_logger(__name__)

def set_wandb_env_vars(cfg):
    """
    Set environment variables from the config dict object.
    The environment variables can be picked up by wandb.
    """
    os.environ["WANDB_ENTITY"] = cfg.get("entity", "")
    os.environ["WANDB_PROJECT"] = cfg.get("project", "")
    os.environ["WANDB_RUN_GROUP"] = cfg.get("group", "")
    os.environ["WANDB_JOB_TYPE"] = cfg.get("job_type", "")
    os.environ["WANDB_NOTES"] = cfg.get("notes", "")
    os.environ["WANDB_TAGS"] = ",".join(cfg.get("tags", ""))


def fix_e(cfg):
    """
    When using "e notation" (1e-5) in a yaml file, it gets interpreted
    as a string rather than a float. This function fixes that.
    """

    def fix(value):
        pattern = r"\d+e\-\d+"
        if re.search(pattern, value):
            return eval(value)
        return value


    for k, v in cfg.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, str):
                    cfg[k][kk] = fix(vv)
        elif isinstance(v, str):
            cfg[k] = fix(v)
    
    return cfg
    
def remove_defaults(cfg):
    """
    Since the yaml file indicates which arguments will
    take default values, this function deletes those arguments
    so that when TrainingArguments is called, all of the deleted
    arguments will get default values.
    """
    to_remove = []
    args = cfg["training_arguments"]
    for key, value in args.items():
        if value == "<default>":
            to_remove.append(key)
    
    for key in to_remove:
        del args[key]

def get_configs(filepath):
    """
    Load config file.
    Returns two dict objects.
    The first has non-TrainingArgument arguments.
    The second has TrainingArgument arguments.
    """
    with open(filepath) as fp:
        cfg = yaml.safe_load(fp)

    
    remove_defaults(cfg)
    cfg = fix_e(cfg)

    # cfg["training_arguments"]["dataloader_num_workers"] = cfg["num_proc"]

    training_args = cfg.pop("training_arguments")
    return cfg, training_args


class NewWandbCB(WandbCallback):
    """
    The current WandbCallback doesn't read some environment variables.
    This implementation fixes that.
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
                    tags=tags if tags is None else tags.split(","),
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