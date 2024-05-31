import torch
from transformers import (
    PreTrainedModel,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    AutoTokenizer,
)


def load_model(path: str) -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(
        path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_auth_token=True,
    ).to("cuda")
    return model


def load_tokenizer(path: str) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
