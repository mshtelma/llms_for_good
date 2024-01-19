# Databricks notebook source
# MAGIC %sh /databricks/python/bin/python -m pip install -r ../requirements.txt --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from huggingface_hub import notebook_login
notebook_login()

# COMMAND ----------

import torch
import peft
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel
from datetime import date

model_name = "llama2-7b"
base_model_name = "meta-llama/Llama-2-7b-chat-hf"
output = f"/dbfs/tmp/rlaif/llm/{model_name}-vegetarian-20240113"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16).to(device)
model = PeftModel.from_pretrained(model, output)
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(output, padding_side="left")

# COMMAND ----------

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id
}

# COMMAND ----------

prompt = "[INST]<<SYS>>You are an AI assistant that specializes in cuisine. Your task is to generate a text related to food preferences, recipes, or ingredients based on the question provided in the instruction. Generate 1 text and do not generate more than 1 text. Be concise and answer within 100 words.<</SYS>> question: What are some protein sources that can be used in dishes?[/INST]"

query = tokenizer.encode(prompt, return_tensors="pt").to(device)
outputs = model.generate(query, **generation_kwargs)
print(tokenizer.decode(outputs[0])[len(prompt) + 6:])

# COMMAND ----------


