import sys
from typing import Optional, Dict, Any

import torch


def get_num_gpus() -> int:
    return torch.cuda.device_count()


def run_inference_mii_process(
    model: str, prompts: list[str], kwargs: Dict[str, Any] = None
) -> list[str]:
    import mii

    if kwargs is None:
        kwargs = {}
    pipe = mii.pipeline(model, all_rank_output=True)
    response = pipe(prompts, **kwargs)
    pipe.destroy()
    return [r.generated_text for r in response]


def run_inference_mii(
    model: str,
    prompts: list[str],
    gpu_count: Optional[int] = None,
    kwargs: Optional[Dict[str, Any]] = None,
) -> list[str]:
    from pyspark.ml.torch.distributor import TorchDistributor

    if gpu_count is None:
        gpu_count = get_num_gpus()
    distributor = TorchDistributor(
        num_processes=gpu_count, local_mode=True, use_gpu=True
    )
    result = distributor.run(
        run_inference_mii_process, model=model, prompts=prompts, kwargs=kwargs
    )
    return result


def run_inference_vllm(
    model: str,
    prompts: list[str],
    gpu_count: Optional[int] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    chunk_size: int = 1000,
):
    from vllm import LLM, SamplingParams

    if gpu_count is None:
        gpu_count = get_num_gpus()
    llm = LLM(model, tensor_parallel_size=gpu_count)
    sampling_params = SamplingParams(**kwargs)
    outputs = []
    for ndx in range(0, len(prompts), chunk_size):
        chunk_idx_max = min(ndx + chunk_size, len(prompts))
        print(f"Processing next chink from {ndx} to {chunk_idx_max}...")
        outputs.extend(llm.generate(prompts[ndx:chunk_idx_max], sampling_params))
    return [r.outputs[0].text for r in outputs]
