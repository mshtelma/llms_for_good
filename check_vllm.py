from vllm import LLM, SamplingParams

prompts = [
    "Mexico is famous for ",
    "The largest country in the world is ",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
print("Creating..")
llm = LLM(model="facebook/opt-125m")
print("Generating..")
responses = llm.generate(prompts, sampling_params)

for response in responses:
    print(response.outputs[0].text)
