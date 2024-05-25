import os
import shutil

import conf
import mlflow
from dataclasses import dataclass, field
from typing import Optional
import math

import torch
from torch.optim import Adam
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    create_reference_model,
    set_seed,
)

from llmsforgood.utils.cli import parse_cmd_args, ScriptArguments
from llmsforgood.utils.datasets import download_dataset, build_dataset
from llmsforgood.utils.inference import run_scoring


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


def run_training(script_args: ScriptArguments):
    mlflow.set_experiment(script_args.mlflow_experiment_path)
    with mlflow.start_run(run_name=os.environ["RUN_NAME"]) as run:
        config = PPOConfig(
            model_name=script_args.model_name,
            learning_rate=script_args.learning_rate,
            project_kwargs={"logging_dir": "/local_disk0/logging_dir"},
            ppo_epochs=script_args.ppo_epochs,
            mini_batch_size=script_args.mini_batch_size,
            batch_size=script_args.batch_size,
            gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        )

        dataset = build_dataset(
            conf.LOCAL_DATASET_PATH, config.model_name, script_args.sample_size
        )

        def collator(data):
            return dict((key, [d[key] for d in data]) for key in data[0])

        # set seed before initializing value head for deterministic eval
        set_seed(config.seed)

        # Now let's build the model, the reference model, and the tokenizer. We first load the model
        # in bfloat16 to save memory using `transformers`.
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_auth_token=True,
        )

        # And then we pass the loaded model to `AutoModelForCausalLMWithValueHead`.
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model  # , peft_config=lora_config
        )

        # We can create a reference model by specifying the number of sharing layers
        # However, since we use LoRA in this demo, we don't need the reference model.
        ref_model = create_reference_model(model, num_shared_layers=20)

        # We make sure to use `Adam` optimizer on the model parameters that require gradients.
        optimizer = Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, padding_side="left"
        )
        tokenizer.pad_token = tokenizer.eos_token

        # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
        ppo_trainer = PPOTrainer(
            config,
            model,
            ref_model=ref_model,  # Set the reference model to None as we are using LoRA
            tokenizer=tokenizer,
            dataset=dataset,
            data_collator=collator,
            optimizer=optimizer,
        )

        # We define the arguments to pass to the `generate` function. These arguments are
        # passed to the `generate` function of the PPOTrainer, which is a wrapper around
        # the `generate` function of the trained model.
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "max_new_tokens": 150,
            "eos_token_id": terminators,
            "pad_token_id": tokenizer.eos_token_id,
        }

        # Base model sometimes generates no response, which causes an error.
        # We overwrite these empty texts with a place holder text.
        place_holder_text = "The base model failed to generate a meaningful text."
        global_setp = 0

        for epoch in range(config.ppo_epochs):
            if ppo_trainer.accelerator.is_main_process:
                print(f"Epoch: {epoch}")
            for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
                global_setp += 1
                query_tensors = batch["input_ids"]
                # Get the response from the base modeld
                response_tensors = []
                for query in query_tensors:
                    response = ppo_trainer.generate(
                        query, return_prompt=False, **generation_kwargs
                    ).squeeze()
                    if not response.shape:
                        response = torch.tensor(
                            tokenizer.encode(place_holder_text)
                        ).squeeze()
                    response_tensors.append(response)
                # Get the score from the reward model
                batch["response"] = [
                    tokenizer.decode(r, skip_special_tokens=True)
                    for r in response_tensors
                ]
                try:
                    scores = run_scoring(batch["response"])
                    rewards_tensors = [
                        torch.tensor(math.log(score / (1.0 - score)))
                        for score in scores
                    ]
                except Exception as e:
                    print(e)
                    scores = [0.5] * config.batch_size
                    rewards_tensors = [torch.tensor(0.0)] * config.batch_size
                # Run PPO step
                stats = ppo_trainer.step(
                    query_tensors, response_tensors, rewards_tensors
                )
                ppo_trainer.log_stats(stats, batch, rewards_tensors)
                for k, v in stats.items():
                    if isinstance(v, (int, float, str, bool)):
                        mlflow.log_metric(k, v, step=global_setp)

                if step % 50 == 0:
                    if ppo_trainer.accelerator.is_main_process:
                        shutil.rmtree(conf.LOCAL_MODEL_PATH, ignore_errors=True)
                        os.makedirs(conf.LOCAL_MODEL_PATH, exist_ok=True)
                        ppo_trainer.save_pretrained(conf.LOCAL_MODEL_PATH)
                        mlflow.log_artifacts(
                            conf.LOCAL_MODEL_PATH, f"checkpoint_{step}"
                        )

                        print(f"STEP: {step}")
                        print(f"PROMPTS: {batch['query']}")
                        print(f"GENERATED: {batch['response']}")
                        print(f"SCORED: {scores}")


if __name__ == "__main__":
    script_args = parse_cmd_args()
    if script_args.run_training:
        run_training(script_args)
    if script_args.download_dataset:
        download_dataset(script_args.dataset_path, conf.LOCAL_DATASET_PATH)
