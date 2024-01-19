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

model_name = "meta-llama/Llama-2-7b-chat-hf"
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

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 200
}

# COMMAND ----------

import pandas as pd
questions = spark.table("rlaif.data.prompts_holdout").toPandas()
display(questions)

# COMMAND ----------

def get_prompt(prompt):
  return f"[INST]<<SYS>>You are an AI assistant that specializes in cuisine. Your task is to generate a text related to food preferences, recipes, or ingredients based on the question provided in the instruction. Generate 1 text and do not generate more than 1 text. Be concise and answer within 100 words.<</SYS>> question: {question}[/INST]"

answers = []

for question in list(questions['prompt'].values):
  prompt = get_prompt(question)
  query = tokenizer.encode(prompt, return_tensors="pt").to(device)
  outputs = model.generate(query, **generation_kwargs)
  answers.append(tokenizer.decode(outputs[0])[len(prompt) + 6:])

# COMMAND ----------

answers = pd.DataFrame(answers).rename(columns={0:"pre_finetuning"})
display(answers)

# COMMAND ----------

df = pd.merge(questions, answers, left_index=True, right_index=True)
df = spark.createDataFrame(df)
display(df)

# COMMAND ----------

df.write.saveAsTable(f"rlaif.data.pre_finetuning")

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
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 200
}

# COMMAND ----------

import pandas as pd
questions = spark.table("rlaif.data.prompts_holdout").toPandas()

# COMMAND ----------

def get_prompt(prompt):
  return f"[INST]<<SYS>>You are an AI assistant that specializes in cuisine. Your task is to generate a text related to food preferences, recipes, or ingredients based on the question provided in the instruction. Generate 1 text and do not generate more than 1 text. Be concise and answer within 100 words.<</SYS>> question: {question}[/INST]"

answers = []

for question in list(questions['prompt'].values):
  prompt = get_prompt(question)
  query = tokenizer.encode(prompt, return_tensors="pt").to(device)
  outputs = model.generate(query, **generation_kwargs)
  answers.append(tokenizer.decode(outputs[0])[len(prompt) + 6:])

# COMMAND ----------

answers = pd.DataFrame(answers).rename(columns={0:"post_finetuning"})
display(answers)

# COMMAND ----------

df = pd.merge(questions, answers, left_index=True, right_index=True)
df = spark.createDataFrame(df)
display(df)

# COMMAND ----------

df.write.saveAsTable("rlaif.data.post_finetuning")

# COMMAND ----------

# MAGIC %md ##Consolidate

# COMMAND ----------

pre_finetune = spark.table("rlaif.data.pre_finetuning").toPandas()
post_finetune = spark.table("rlaif.data.post_finetuning").toPandas()
df = pd.merge(pre_finetune, post_finetune, on=["prompt"])
display(df)

# COMMAND ----------


