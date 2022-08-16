from dataclasses import dataclass
from typing import Dict, List, Union

@dataclass
class DataConfig:
    dataset_name: str
    dataset_config_name: str
    max_seq_length: int
    # set to small number for debugging, -1 for all rows
    n_rows: int

@dataclass
class ModelConfig:
    model_name_or_path: str
    config_name: str # Pretrained config name or path if not the same as model_name
    tokenizer_name: str # Pretrained tokenizer name or path if not the same as model_name"
    model_revision: str
    use_auth_token: Union[bool, str] # pass token as string or True if logged in
    from_tf: bool # set to 'yes' if loading from tensorflow weights
    from_flax: bool # set to 'yes' if loading from flax weights

@dataclass
class TrainingArgumentsConfig:
    # TrainingArguments
    # https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments


    # output
    overwrite_output_dir: bool

    # training
    do_train: bool
    resume_from_checkpoint: str

    # hyperparams
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    group_by_length: bool
    learning_rate: float
    weight_decay: float
    seed: int

    # schedule + steps
    num_train_epochs: int
    lr_scheduler_type: str 
    warmup_ratio: float
    warmup_steps: int
    max_steps: int

    # optimizer
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    max_grad_norm: float
    optim: str
    adafactor: bool

    # logging
    log_level: str
    log_level_replica: str
    log_on_each_node: bool
    logging_dir: str
    logging_strategy: str
    logging_first_step: bool
    logging_steps: int
    logging_nan_inf_filter: bool

    # saving
    save_strategy: str
    save_steps: int
    save_total_limit: int

    # dtype
    fp16: bool
    bf16: bool # bf16 requires Ampere GPUs or newer (A100, A6000, rtx 3080)
    fp16_opt_level: str
    half_precision_backend: str
    bf16_full_eval: bool
    fp16_full_eval: bool
    tf32: bool

    # evaluation/prediction
    do_eval: bool
    evaluation_strategy: str
    eval_delay: int
    include_inputs_for_metrics: bool
    do_predict: bool
    jit_mode_eval: bool

    # hub
    hub_model_id: str
    hub_token: str

    # rarely used
    debug: str
    prediction_loss_only: bool
    eval_accumulation_steps: int
    use_ipex: bool
    save_on_each_node: bool
    no_cuda: bool
    use_mps_device: bool
    data_seed: int
    local_rank: int
    xpu_backend: str
    tpu_num_cores: int
    tpu_metrics_debug: bool
    dataloader_drop_last: bool
    past_index: int
    run_name: str
    disable_tqdm: bool
    remove_unused_columns: bool
    label_names: List[str]
    greater_is_better: bool
    ignore_data_skip: bool
    sharded_ddp: str
    fsdp: str
    fsdp_min_num_params: int
    fsdp_transformer_layer_cls_to_wrap: str
    deepspeed: Union[Dict, str]
    label_smoothing_factor: float
    length_column_name: str
    ddp_find_unused_parameters: bool
    ddp_bucket_cap_mb: int
    skip_memory_metrics: bool
    use_legacy_prediction_loop: bool
    # deprecated
    per_gpu_train_batch_size: int
    per_gpu_eval_batch_size: int
    fp16_backend: str 
    push_to_hub_model_id: str
    push_to_hub_organization: str
    push_to_hub_token: str
    # _n_gpu: 
    mp_parameters: str
    auto_find_batch_size: bool
    full_determinism: bool
    torchdynamo: bool
    ray_scope: str

@dataclass
class MlflowConfig:
    log_artifact: bool
    experiment_name: str
    tags: List[str]
    test: bool
    model_type: str
    nested_run: bool
    run_id: str
    flatten_params: bool
    tracking_uri: str

@dataclass
class WandbConfig:
    project: str
    entity: str
    group: str
    tags: List[str]
    notes: str
    job_type: str