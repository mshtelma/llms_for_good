# Databricks notebook source
# MAGIC %md
# MAGIC ## Pre fine-tuned model

# COMMAND ----------

from huggingface_hub import notebook_login
notebook_login()

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

import pandas as pd

questions = spark.table("rlaif.data.prompts_holdout").toPandas()
display(questions)

# COMMAND ----------

def prompt_generate(text):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an AI assistant that specializes in cuisine. Your task is to generate a text related to food preferences, recipes, or ingredients based on the question provided below. Generate 1 text and do not generate more than 1 text. Be concise and use no more than 100 words.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

answers = []

for question in list(questions["prompt"].values):
    prompt = prompt_generate(question)
    query = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(query, **generation_kwargs)
    response = outputs[0][query.shape[-1]+1:]
    answers.append(tokenizer.decode(response, skip_special_tokens=True))

# COMMAND ----------

answers = pd.DataFrame(answers).rename(columns={0: "pre_finetuning"})
display(answers)

# COMMAND ----------

df = pd.merge(questions, answers, left_index=True, right_index=True)
df = spark.createDataFrame(df)
display(df)

# COMMAND ----------

df.write.saveAsTable(f"rlaif.data.pre_finetuning_llama3")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Post fine-tuned model

# COMMAND ----------

import torch
import gc

gc.collect()
torch.cuda.empty_cache()
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %sh /databricks/python/bin/python -m pip install -r requirements.txt --quiet

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

import pandas as pd

questions = spark.table("rlaif.data.prompts_holdout").toPandas()

# COMMAND ----------

def prompt_generate(text):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an AI assistant that specializes in cuisine. Your task is to generate a text related to food preferences, recipes, or ingredients based on the question provided below. Generate 1 text and do not generate more than 1 text. Be concise and use no more than 100 words.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

answers = []

for question in list(questions["prompt"].values):
    prompt = prompt_generate(question)
    query = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(query, **generation_kwargs)
    response = outputs[0][query.shape[-1]+1:]
    answers.append(tokenizer.decode(response, skip_special_tokens=True))

# COMMAND ----------

answers = pd.DataFrame(answers).rename(columns={0: "post_finetuning"})
display(answers)

# COMMAND ----------

df = pd.merge(questions, answers, left_index=True, right_index=True)
df = spark.createDataFrame(df)
display(df)

# COMMAND ----------

df.write.saveAsTable("rlaif.data.post_finetuning_llama3")

# COMMAND ----------

# MAGIC %md ##Consolidate

# COMMAND ----------

pre_finetune = spark.table("rlaif.data.pre_finetuning_llama3").toPandas()
post_finetune = spark.table("rlaif.data.post_finetuning_llama3").toPandas()
df = pd.merge(pre_finetune, post_finetune, on=["prompt"])
display(df)
