import os
from pathlib import Path

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import DatabricksError
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer


def download_dataset(dbfs_path: str, local_path: str) -> None:
    w = WorkspaceClient()
    files = w.files.list_directory_contents(dbfs_path)
    local_path = Path(local_path)
    local_path.mkdir(parents=True, exist_ok=True)

    for entry in files:
        local_file_path = local_path / entry.name
        local_file_path_tmp = Path(f"{str(local_file_path.absolute())}.tmp")
        try:
            with w.files.download(entry.path).contents as response:
                with open(str(local_file_path_tmp), "wb") as f:
                    # Download data in chunks to avoid memory issues.
                    for chunk in iter(lambda: response.read(64 * 1024 * 1024), b""):
                        f.write(chunk)
        except DatabricksError as e:
            if e.error_code == "REQUEST_LIMIT_EXCEEDED":
                raise Exception(f"Too many concurrent download operations!") from e
            if e.error_code == "NOT_FOUND":
                raise FileNotFoundError(f" {entry.path} not found.") from e
            raise e
        local_file_path_tmp.rename(local_file_path)


def prompt_generate(text):
    return f"""<|start_header_id|>system<|end_header_id|>
    You are an AI assistant that specializes in cuisine. Your task is to generate a text related to food preferences, recipes, or ingredients based on the question provided below. Generate 1 text and do not generate more than 1 text. Be concise and use no more than 100 words.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


def build_dataset(config, dataset_path):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_path (`str`):
            The path to the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_from_disk(dataset_path)

    # if script_args.sample_size:
    #    ds = ds.select(range(script_args.sample_size))

    def tokenize(sample):
        prompt = prompt_generate(sample["prompt"])
        sample["input_ids"] = tokenizer.encode(prompt)
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")

    return ds
