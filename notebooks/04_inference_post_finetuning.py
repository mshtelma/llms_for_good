# Databricks notebook source
# MAGIC %sh /databricks/python/bin/python -m pip install -r ../requirements.txt --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Authenticate against Hugging Face
import os
from huggingface_hub import login

os.environ["HF_TOKEN"] = dbutils.secrets.get("rlaif", "hf_token")
login(os.environ["HF_TOKEN"])

# COMMAND ----------

import torch
import peft
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel
from datetime import date

today = date.today().strftime("%Y%m%d")
model_name = "llama3-8b-vegetarian"
base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
output = f"/dbfs/rlaif/llm/{model_name}-{today}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name, torch_dtype=torch.bfloat16
).to(device)
model = PeftModel.from_pretrained(model, output)
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(output, padding_side="left")

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
    Question: {text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

query = tokenizer.encode(prompt_generate(text), return_tensors="pt").to(device)
outputs = model.generate(query, **generation_kwargs)
response = outputs[0][query.shape[-1]+1:]
print(tokenizer.decode(response, skip_special_tokens=True))

# COMMAND ----------


