import logging
import os
import shutil
import sys
from dataclasses import field, dataclass
from typing import Optional

import datasets
import numpy as np
import transformers
from accelerate import Accelerator
from accelerate.utils import broadcast, broadcast_object_list

from peft import LoraConfig
from torch.optim.lr_scheduler import CosineAnnealingLR

# from peft import LoraConfig
# from torch.optim.lr_scheduler import ExponentialLR

import conf
import mlflow
import math

import torch
from torch.optim import Adam
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import (
    AutoModelForCausalLMWithValueHead,
    DPOTrainer,
    DPOConfig,
    create_reference_model,
    set_seed,
)
from trl import DPOConfig, DPOTrainer


from llmsforgood.utils.inference import run_reward_scoring
from llmsforgood.utils.lion import Lion
from llmsforgood.utils.cli import parse_cmd_args, ScriptArguments
from llmsforgood.utils.datasets import (
    download_dataset,
    build_dataset_with_prompts,
    build_question_answer_dataset,
)

logger = logging.getLogger(__name__)

########################################################################
# This is a fully working simple example to use trl with accelerate.
#
# This example fine-tunes Meta-Llama-3-8B-Instruct to generate more vegetarian contents
# by using prompts dataset. We use PPO (proximal policy optimization)
# to optimize the model.
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - multi GPUS (using DeepSpeed ZeRO-Offload stages 1 & 2)
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, first initialize the accelerate
# configuration with `accelerate config`
#
########################################################################


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    model_run_id: Optional[str] = field(
        default=None,
        metadata={"help": "Run ID containing checkpoint to tune using DPO"},
    )
    model_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Artifact path for the checkpoint to tune using DPO"},
    )

    train: Optional[bool] = field(
        default=False,
        metadata={"help": "Run training."},
    )
    download_dataset: Optional[bool] = field(
        default=False,
        metadata={"help": "Download dataset."},
    )
    download_model: Optional[bool] = field(
        default=False,
        metadata={"help": "Download model from MLflow."},
    )
    dataset_path: Optional[str] = field(
        default="/Volumes/msh/rlaif/data/hf_train_dataset",
        metadata={"help": "the path to the training dataset"},
    )
    mlflow_experiment_path: Optional[str] = field(
        default="/Shared/llm4good_trl",
        metadata={"help": "MLflow Experiment path"},
    )

    # data parameters
    beta: Optional[float] = field(
        default=0.1, metadata={"help": "the beta parameter for DPO loss"}
    )

    learning_rate: Optional[float] = field(
        default=5e-6, metadata={"help": "optimizer learning rate"}
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine", metadata={"help": "the lr scheduler type"}
    )
    warmup_steps: Optional[int] = field(
        default=100, metadata={"help": "the number of warmup steps"}
    )
    weight_decay: Optional[float] = field(
        default=0.05, metadata={"help": "the weight decay"}
    )
    optimizer_type: Optional[str] = field(
        default="paged_adamw_32bit", metadata={"help": "the optimizer type"}
    )

    per_device_train_batch_size: Optional[int] = field(
        default=2, metadata={"help": "train batch size per device"}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=1, metadata={"help": "eval batch size per device"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    gradient_checkpointing_use_reentrant: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to use reentrant for gradient checkpointing"},
    )

    max_prompt_length: Optional[int] = field(
        default=512, metadata={"help": "the maximum prompt length"}
    )
    max_length: Optional[int] = field(
        default=1024, metadata={"help": "the maximum sequence length"}
    )
    max_steps: Optional[int] = field(
        default=1000, metadata={"help": "max number of training steps"}
    )
    logging_steps: Optional[int] = field(
        default=10, metadata={"help": "the logging frequency"}
    )
    save_steps: Optional[int] = field(
        default=100, metadata={"help": "the saving frequency"}
    )
    eval_steps: Optional[int] = field(
        default=100, metadata={"help": "the evaluation frequency"}
    )

    output_dir: Optional[str] = field(
        default="./results", metadata={"help": "the output directory"}
    )
    log_freq: Optional[int] = field(
        default=1, metadata={"help": "the logging frequency"}
    )
    load_in_4bit: Optional[bool] = field(
        default=True, metadata={"help": "whether to load the model in 4bit"}
    )

    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )


def run_training(script_args: ScriptArguments):
    mlflow.set_experiment(script_args.mlflow_experiment_path)

    dataset = build_question_answer_dataset(conf.LOCAL_DATASET_PATH)
    dataset_dict = dataset.train_test_split(0.01)

    accelerator = Accelerator()
    model_path = ""
    if accelerator.is_local_main_process:
        model_path = mlflow.artifacts.download_artifacts(
            run_id=script_args.model_run_id,
            artifact_path=script_args.model_checkpoint,
            dst_path=conf.LOCAL_MODEL_PATH,
        )
        print(f"Model path: {model_path}")
        model_path = broadcast_object_list([model_path])[0]
    print(model_path)

    dpo_config = DPOConfig(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        eval_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name="dpo_llama2",
        gradient_checkpointing_kwargs=dict(
            use_reentrant=script_args.gradient_checkpointing_use_reentrant
        ),
        max_length=script_args.max_length,
        max_prompt_length=script_args.max_prompt_length,
    )

    # set seed before initializing value head for deterministic eval
    set_seed(45)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_auth_token=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    dpo_trainer = DPOTrainer(
        model,
        args=dpo_config,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["test"],
        tokenizer=tokenizer,
    )

    with mlflow.start_run() as run:
        dpo_trainer.train()
        dpo_trainer.save_model("/tmp")

        save_checkpoint(dpo_trainer, run, "final")


def save_checkpoint(dpo_trainer, run, step):
    checkpoint_path = os.path.join(conf.LOCAL_MODEL_PATH, "checkpoint")
    shutil.rmtree(checkpoint_path, ignore_errors=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    dpo_trainer.model.save_pretrained(checkpoint_path)
    mlflow.log_artifacts(
        checkpoint_path,
        f"checkpoint_{step}",
        run_id=run.info.run_id,
    )


if __name__ == "__main__":
    os.environ["MLFLOW_TRACKING_URI"] = "databricks"
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    if script_args.train:
        run_training(script_args)
    if script_args.download_dataset:
        download_dataset(script_args.dataset_path, conf.LOCAL_DATASET_PATH)
