import torch
from peft import PeftModel
from transformers import (
    PreTrainedModel,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    AutoTokenizer,
    AutoConfig,
)


def has_adapter(config):
    adapter_attributes = ["adapter_config", "adapter_fusion_config", "adapter_list"]
    return any(hasattr(config, attr) for attr in adapter_attributes)


def load_model(path: str) -> PreTrainedModel:
    config = AutoConfig.from_pretrained(path)

    model = AutoModelForCausalLM.from_pretrained(
        path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_auth_token=True,
    ).to("cuda")

    if has_adapter(config):
        model = PeftModel.from_pretrained(model, path)
        model = model.merge_and_unload()
    return model


def load_tokenizer(path: str) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
