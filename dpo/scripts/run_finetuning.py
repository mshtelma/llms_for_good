# Databricks notebook source
# %sh /databricks/python/bin/python -m pip install -r ../requirements.txt --quiet

# COMMAND ----------

# MAGIC %sh /databricks/python/bin/python -m pip install -r ./requirements.txt --quiet

# COMMAND ----------

# MAGIC %pip install aiohttp[speedups] --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from datetime import date
today = date.today().strftime("%Y%m%d")
model_name = "llama2-7b"
base_model_name = "meta-llama/Llama-2-7b-chat-hf"
base_path = "/dbfs/tmp/alexm/llm/stack_llama_2/"
dataset_path = f"{base_path}/data/vegetarian_data.json"
# dataset_path = "/dbfs/tmp/alexm/llm/stack_llama_2/data/"
output = f"{base_path}/{model_name}-vegetarian-{today}"
tb_output = f"{base_path}/tb/{model_name}-vegetarian-{today}"
logdir = "/databricks/driver/logdir/dpo"

# COMMAND ----------

# spark.table("ai_blog.gen_data.vegetarian_questions").write.format("json").save("dbfs:/tmp/alexm/llm/stack_llama_2/data/vegetarian_data.json")

# COMMAND ----------

# display(spark.read.format("json").load("dbfs:/tmp/alexm/llm/stack_llama_2/data/vegetarian_data.json"))

# COMMAND ----------

# from datetime import date
# today = date.today().strftime("%Y%m%d")
# model_name = "llama2-7b"
# base_model_name = "meta-llama/Llama-2-7b-chat-hf"
# dataset_path = "/dbfs/tmp/rlaif/data/"
# output = f"/dbfs/tmp/rlaif/llm/{model_name}-vegetarian-{today}"
# tb_output = f"/dbfs/tmp/rlaif/tb/{model_name}-vegetarian-{today}"
# logdir = "/databricks/driver/logdir/trl"

# COMMAND ----------

!mkdir -p {output}

# COMMAND ----------

!mkdir -p {tb_output}

# COMMAND ----------

import os
os.environ['SCRIPT'] = "./sft_llama2.py" 
os.environ['OUTPUT'] = output
os.environ['TB_OUTPUT'] = tb_output
os.environ['DATSET_PATH'] = dataset_path
os.environ['LOGDIR'] = logdir

# COMMAND ----------

# import os
# os.environ['SCRIPT'] = "../llmsforgood/llama2-7b-vegi.py" 
# os.environ['OUTPUT'] = output
# os.environ['TB_OUTPUT'] = tb_output
# os.environ['DATSET_PATH'] = dataset_path
# os.environ['LOGDIR'] = logdir

# COMMAND ----------

from tensorboard import notebook
notebook.start("--logdir {} --reload_multifile True".format(logdir))

# COMMAND ----------

# To kill a tensorboard process
from tensorboard import notebook
notebook.list()

# COMMAND ----------

from huggingface_hub import notebook_login

notebook_login()

# COMMAND ----------

# /Workspace/Repos/alex.miller@databricks.com/llms_for_good/dpo/scripts/config/deepspeed_zero1.yaml

# COMMAND ----------

db_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ['DATABRICKS_HOST'] = db_host
os.environ['DATABRICKS_TOKEN'] = db_token

# COMMAND ----------

# MAGIC %sh accelerate launch --config_file ./config/deepspeed_zero1.yaml $SCRIPT \
# MAGIC     --output_dir="./sft" \
# MAGIC     --max_steps=500 \
# MAGIC     --logging_steps=10 \
# MAGIC     --save_steps=10 \
# MAGIC     --per_device_train_batch_size=4 \
# MAGIC     --per_device_eval_batch_size=1 \
# MAGIC     --bf16=True \
# MAGIC     --learning_rate=1e-4 \
# MAGIC     --report_to="none" \
# MAGIC     --model_save_path=$OUTPUT
# MAGIC     # --log_with tensorboard \
# MAGIC     # --gradient_checkpointing=False \
# MAGIC     # --group_by_length=False \
# MAGIC     # --learning_rate=1e-4 \
# MAGIC     # --lr_scheduler_type="cosine" \
# MAGIC     # --warmup_steps=100 \
# MAGIC     # --weight_decay=0.05 \
# MAGIC     # --optim="paged_adamw_32bit" \
# MAGIC     # --remove_unused_columns=False \
# MAGIC     # --run_name="sft_llama2" 

# COMMAND ----------

# MAGIC %sh cp -r $LOGDIR $TB_OUTPUT

# COMMAND ----------

!ls -lah {output}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the model to mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from huggingface_hub import notebook_login
notebook_login()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log and register the model to mlflow

# COMMAND ----------

import pandas as pd
import numpy as np
import transformers
import mlflow
import torch
import peft
import accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM

class vegetarian(mlflow.pyfunc.PythonModel):
    def __init__(self, model_name):
        self.model_name = model_name
    
    def load_context(self, context):
        """
        This method initializes the tokenizer and language model
        using the specified model repository.
        """
        # Initialize tokenizer and language model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16).to(self.device)
        self.model = peft.PeftModel.from_pretrained(self.model, context.artifacts['repository'])
        self.model = self.model.merge_and_unload()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(context.artifacts['repository'])
        self.generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id
        }

    def get_prompt(self, input_text):
        return f"[INST]<<SYS>>You are an AI assistant that specializes in cuisine. Your task is to generate a text related to food preferences, recipes, or ingredients based on the question provided in the instruction. Generate 1 text and do not generate more than 1 text. Be concise and answer within 100 words.<</SYS>> question: {input_text} [/INST]"
    
    def predict(self, context, model_input):
        """
        This method generates prediction for the given input.
        """
        generated_text = []
        for index, row in model_input.iterrows():
          input_text = row["input"]
          prompt = self.get_prompt(input_text)
          query = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
          outputs = self.model.generate(query, **self.generation_kwargs)
          generated_text.append(self.tokenizer.decode(outputs[0])[len(prompt) + 6:])

        return pd.Series(generated_text)

# COMMAND ----------

from datetime import date
today = date.today().strftime("%Y%m%d")
model_name = "llama2-7b"
base_model_name = "meta-llama/Llama-2-7b-chat-hf"
output = f"/dbfs/tmp/rlaif/llm/{model_name}-vegetarian-{today}"

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

# Define input and output schema
input_schema = Schema([ColSpec(DataType.string, "input")])
output_schema = Schema([ColSpec(DataType.string)])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
input_example=pd.DataFrame({"input":["Give me a recipe for a healthy smoothie."]})

# Log the model with its details such as artifacts, pip requirements and input example
# This may take about 1.2 minutes to complete
torch_version = torch.__version__.split("+")[0]

with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        "model",
        python_model=vegetarian(base_model_name),
        artifacts={'repository' : output},
        pip_requirements=[
            f"torch=={torch_version}", 
            f"transformers=={transformers.__version__}", 
            f"accelerate=={accelerate.__version__}",
            f"peft=={peft.__version__}"
            ],
        input_example=input_example,
        signature=signature
    )

# COMMAND ----------

# Register model
import mlflow
mlflow.set_registry_uri('databricks-uc')
registered_name = f"rlaif.model.vegetarian-{model_name}"
result = mlflow.register_model(
    "runs:/" + run.info.run_id + "/model",
    registered_name,
    await_registration_for=1000,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the logged model
# MAGIC
# MAGIC Restart the Python to release the GPU memory occupied in Training.

# COMMAND ----------

import mlflow
import pandas as pd

model_name = "llama2-7b"
registered_name = f"rlaif.model.vegetarian-{model_name}"
logged_model = f"models:/{registered_name}/1"

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# COMMAND ----------

# Predict on a Pandas DataFrame.
questions = pd.DataFrame({"input":[
  "Give me a recipe for a healthy smoothie?", 
  "Where can I find the best breakfast for lazy Sunday morning?", 
  "Tell me some ingredients for protein-rich, healthy lunch?"]})
answers = loaded_model.predict(questions)

# COMMAND ----------

for index, answer in enumerate(answers):
  question = questions['input'][index]
  print(index)
  print(f"Question: {question}")
  print(f"Answer: {answer}\n")

# COMMAND ----------

from mlflow.tracking.client import MlflowClient
mlflow.set_registry_uri('databricks-uc')
MlflowClient().set_registered_model_alias(registered_name, "champion", 1)

# COMMAND ----------


