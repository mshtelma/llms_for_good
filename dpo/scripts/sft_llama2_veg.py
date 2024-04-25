# Fine-Tune Llama2-7b on SE paired dataset
import os
from dataclasses import dataclass, field
from typing import Optional, List, Union

import mlflow

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments

from trl import SFTTrainer
from trl.import_utils import is_npu_available, is_xpu_available
from trl.trainer import ConstantLengthDataset

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})
    log_with: Optional[str] = field(default="tensorboard", metadata={"help": "use 'tensorboard' to log with tensorboard"})
    dataset_name: Optional[str] = field(default="lvwerra/stack-exchange-paired", metadata={"help": "the dataset name"})
    subset: Optional[str] = field(default="data/finetune", metadata={"help": "the subset to use"})
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    size_valid_set: Optional[int] = field(default=4000, metadata={"help": "the size of the validation set"})
    streaming: Optional[bool] = field(default=True, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    seq_length: Optional[int] = field(default=1024, metadata={"help": "the sequence length"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})
    packing: Optional[bool] = field(default=True, metadata={"help": "whether to use packing for SFTTrainer"})
    model_save_path: Optional[str] = field(
        default="/dbfs/tmp/alexm/llm/stack_llama_2/",
        metadata={"help": "the path to save the model"},
    )
    databricks_host: Optional[str] = field(default="host", metadata={"help": "Databricks host environment variable"})
    databricks_token: Optional[str] = field(default="token", metadata={"help": "Databricks token environment variable"})

    # LoraConfig
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})


# Initialize the HfArgumentParser with both your ScriptArguments and TrainingArguments
parser = HfArgumentParser((ScriptArguments, TrainingArguments))

# Parse the arguments into instances of your data classes
script_args, training_args = parser.parse_args_into_dataclasses()

os.environ['DATABRICKS_HOST'] = script_args.databricks_host
os.environ['DATABRICKS_TOKEN'] = script_args.databricks_token
# mlflow.set_tracking_uri("databricks")
# mlflow.set_experiment("/Users/alex.miller@databricks.com/dpo-sft-llama2")

target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head']
target_modules = ["q_proj", "v_proj"]
peft_config = LoraConfig(
    r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    target_modules=target_modules,
    bias="none",
    task_type="CAUSAL_LM",
)

if training_args.group_by_length and script_args.packing:
    raise ValueError("Cannot use both packing and group by length")

# `gradient_checkpointing` was True by default until `1f3314`, but it's actually not used.
# `gradient_checkpointing=True` will cause `Variable._execution_engine.run_backward`.
if training_args.gradient_checkpointing:
    raise ValueError("gradient_checkpointing not supported")


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    text = f"Question: {example['prompt']}\n\nAnswer: {example['good_answer']}"
    return text


def create_datasets(tokenizer, args):
    dataset = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split=args.split,
        use_auth_token=True,
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,
    )
    if args.streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=None)
    else:
        dataset = dataset.train_test_split(test_size=0.005, seed=None)
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
print(f"Loading base model: {script_args.model_name}")
base_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    use_auth_token=True,
)
# base_model = AutoModelForCausalLM.from_pretrained(
#     script_args.model_name,
#     quantization_config=bnb_config,
#     device_map={"": Accelerator().local_process_index},
#     trust_remote_code=True,
#     use_auth_token=True,
# )
base_model.config.use_cache = False

print(f"Loading tokenizer: {script_args.model_name}")
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

train_dataset, eval_dataset = create_datasets(tokenizer, script_args)

trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    packing=script_args.packing,
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_args,
)
trainer.train()
# trainer.save_model(training_args.output_dir)
trainer.save_model(training_args.model_save_path)

output_dir = os.path.join(training_args.model_save_path, "final_checkpoint")
trainer.model.save_pretrained(training_args.model_save_path)

# Free memory for merging weights
del base_model
if is_xpu_available():
    torch.xpu.empty_cache()
elif is_npu_available():
    torch.npu.empty_cache()
else:
    torch.cuda.empty_cache()

model = AutoPeftModelForCausalLM.from_pretrained(training_args.model_save_path, device_map="auto", torch_dtype=torch.bfloat16)
model = model.merge_and_unload()

output_merged_dir = os.path.join(training_args.model_save_path, "final_merged_checkpoint")
model.save_pretrained(output_merged_dir, safe_serialization=True)
