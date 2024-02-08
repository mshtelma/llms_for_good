import json
import os
import pathlib
import shutil
import random
from typing import Dict, Any, Union

from llmsforgood.prompt_generation.inference_mii import (
    run_inference_mii,
    run_inference_vllm,
)

import re


def extract_json(result: str) -> str:
    return re.search(r"\{.*\}", result, re.DOTALL).group()


def clean_string(str_variable: str) -> str:
    split_str = str_variable.replace("\n", "").split()
    return " ".join(split_str)


def process_result(result: str) -> Union[Dict[str, Any], None]:
    try:
        json_part = extract_json(result)
        clean_str = clean_string(json_part)
        return json.loads(clean_str)
    except:
        return None


def download_model(model_name: str, local_dir: str = "/local_disk0/model") -> str:
    import mlflow

    if local_dir:
        model_dir = pathlib.Path(local_dir) / "model"
        if model_dir.exists():
            return str(model_dir)
        pathlib.Path(local_dir).mkdir(exist_ok=True)
    mlflow.set_registry_uri("databricks-uc")
    local_path = mlflow.artifacts.download_artifacts(model_name, dst_path=local_dir)
    print(f"Downloaded model to {local_path}")
    shutil.copytree(
        os.path.join(local_path, "components/tokenizer"),
        os.path.join(local_path, "model"),
        dirs_exist_ok=True,
    )
    print(f"Downloaded model to {local_path}")
    return os.path.join(local_path, "model")


def format_prompt(sys_prompt: str, prompt: str):
    return f"[INST]<<SYS>>" f"{sys_prompt}" f"<</SYS>> " f"{prompt} [/INST]"


def generate_questions_for_topics(
    model_path: str,
    system_prompt: str,
    prompt: str,
    topic_list: list[str],
    num_topics: int = 2,
    number_of_questions_to_generate: int = 100,
    multiplier: float = 1.05,
    engine: str = "mii",
    kwargs: Dict[str, Any] = None,
    chunk_size: int = 1000,
):
    questions = [
        f"{prompt}: {', '.join(random.sample(topic_list, num_topics))}."
        for _ in range(int(number_of_questions_to_generate * multiplier))
    ]
    prompts = [format_prompt(system_prompt, q) for q in questions]
    if engine == "mii":
        generated_questions_text = run_inference_mii(model_path, prompts, kwargs=kwargs)
    elif engine == "vllm":
        generated_questions_text = run_inference_vllm(
            model_path, prompts, kwargs=kwargs, chunk_size=chunk_size
        )
    else:
        raise Exception(f"Unsupported engine: {engine}! Supported engines are: [mii].")
    print(generated_questions_text)
    parsed_generated_questions = [process_result(r) for r in generated_questions_text]
    parsed_generated_questions = [r for r in parsed_generated_questions if r]
    parsed_generated_questions = parsed_generated_questions[
        :number_of_questions_to_generate
    ]
    return parsed_generated_questions
