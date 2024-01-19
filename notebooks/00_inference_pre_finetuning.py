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

# COMMAND ----------

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 125
}

# COMMAND ----------

prompt = "[INST]<<SYS>>You are an AI assistant that specializes in cuisine. Your task is to generate a text related to food preferences, recipes, or ingredients based on the question provided in the instruction. Generate 1 text and do not generate more than 1 text. Be concise and answer within 100 words.<</SYS>> question: What are some protein sources that can be used in dishes?[/INST]"

query = tokenizer.encode(prompt, return_tensors="pt").to(device)
outputs = model.generate(query, **generation_kwargs)
print(tokenizer.decode(outputs[0])[len(prompt) + 6:])

# COMMAND ----------


