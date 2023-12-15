# Databricks notebook source


# COMMAND ----------

# MAGIC %sh /databricks/python/bin/python -m pip install -r ../../../requirements.txt --quiet
# MAGIC %sh /databricks/python/bin/python -m pip install trl peft

# COMMAND ----------

# MAGIC %sh cd ../../.. && /databricks/python/bin/python -m pip install -e . --quiet

# COMMAND ----------

# MAGIC %pip install aiohttp[speedups] --quiet


# COMMAND ----------


from huggingface_hub import notebook_login

# Login to Huggingface to get access to the model
notebook_login()

# COMMAND ----------
# MAGIC %sh accelerate launch  --config_file ../yamls/accelerate/zero2.yaml ../llmsforgood/llama2-7b-vegi.py
