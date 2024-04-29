# Databricks notebook source
from huggingface_hub import notebook_login
notebook_login()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference
# MAGIC The example in the model card should also work on Databricks with the same environment.

# COMMAND ----------

# Load model to text generation pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    use_auth_token=True,
).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

# COMMAND ----------

prompts = spark.read.csv("/rlaif/data/prompts.csv", header=True)
display(prompts)

# COMMAND ----------

# MAGIC %md 
# MAGIC -----------------------------------------

# COMMAND ----------

place_holder_text = "The base model failed to generate a meaningful text."
response_tensor = tokenizer.encode(place_holder_text)
response_tensor = torch.tensor(response_tensor)
response_tensor = response_tensor.squeeze()
tokenizer.decode(response_tensor.squeeze(), skip_special_tokens=True)

# COMMAND ----------

r = torch.tensor([128000, 128000, 128006,   9125, 128007,   7361,    262,   1472,    527,
           459,  15592,  18328,    430,  46672,    304,  36105,     13,   4718,
          3465,    374,    311,   7068,    264,   1495,   5552,    311,   3691,
         19882,     11,  19141,     11,    477,  14293,   3196,    389,    279,
          3488,   3984,   3770,     13,  20400,    220,     16,   1495,    323,
           656,    539,   7068,    810,   1109,    220,     16,   1495,     13,
          2893,  64694,    323,   1005,    912,    810,   1109,    220,   1041,
          4339,     13, 128009, 128006,    882, 128007,   7361,    262,  16225,
            25,   3053,    499,   7079,   1063,   8753,  53161,  46895,    273,
         26863,    430,    649,    387,  10235,   6288,    323,   6847,   7801,
        128009, 128006,  78191, 128007])
print(r.squeeze())
print(r.squeeze().squeeze())
#tokenizer.decode(r.squeeze(), skip_special_tokens=True)

# COMMAND ----------

r = torch.tensor(128009)
print(r, r.shape)
r = r.squeeze()
print(r, r.shape)
#r = r.reshape(1)
#print(r, r.shape)
tokenizer.decode(r.squeeze(), skip_special_tokens=True)

# COMMAND ----------

r = torch.tensor(128009)
if r.shape:
  print('!')

# COMMAND ----------

place_holder_text = "The base model failed to generate a meaningful text."
response_tensor = tokenizer.encode(place_holder_text)
print(response_tensor)

# COMMAND ----------

!ls -ltr /dbfs/rlaif/llm/llama3-8b-vegetarian-20240429

# COMMAND ----------

[torch.tensor(0.)] * 4

# COMMAND ----------

df = spark.read.csv("/rlaif/data/prompts.csv")
display(df)

# COMMAND ----------

# MAGIC %md 
# MAGIC -----------------------------------------

# COMMAND ----------

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
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

# COMMAND ----------

text = "What are some protein sources that can be used in dishes?"

def prompt_generate(text):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an AI assistant that specializes in cuisine. Your task is to generate a text related to food preferences, recipes, or ingredients based on the question provided below. Generate 1 text and do not generate more than 1 text. Be concise and use no more than 100 words.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

query = tokenizer.encode(prompt_generate(text), return_tensors="pt").to(device)
outputs = model.generate(query, **generation_kwargs)
response = outputs[0][query.shape[-1]+1:]
print(tokenizer.decode(response, skip_special_tokens=True))

# COMMAND ----------


