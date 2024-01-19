# Databricks notebook source
# MAGIC %pip install git+https://github.com/mlflow/mlflow.git@gateway-migration --quiet
# MAGIC %pip install aiohttp[speedups] --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use Llama2 via Foundation Models API to generate hold out dataset for evaluation

# COMMAND ----------

import re
import json
import random
from mlflow.deployments import get_deploy_client

deploy_client = get_deploy_client("databricks")
topic_list = ["Nutritious", "Plant-Based", "Meal Planning", "Cooking Techniques", "Vegetarianism",    
          "Global Dishes", "Seasonal Recipes", "Kids' Meals", "Vegan", "Environmental Impact",
          "Diet Myths", "Special Diets", "Dining Out", "Athlete Nutrition", "Homemade Snacks", 
          "Budget-Friendly", "Wine Pairing", "Different Cultures", "Bodybuilding", "Holiday Recipes",
          "Exotic Cuisine", "High Calorie", "Healthy Food", "Low Cost", "Fresh Ingredience",
          "Mediterranean", "Indian", "Asian", "African", "South American",
          "Popular", "Fine Dining", "Table Manner", "Michelin Star", "French",
          "Bread", "Noodles", "Healthy", "Unhealthy", "Substantial",
          "Culinary Diversity", "Innovative Dish", "Fusion", "Seasonal", "Tasting Menu",
          "Herbs", "Homestyle", "Organic", "Locally Sourced", "Farm-to-Table",
          "Heirloom", "Spicy", "Authentic Flavors", "Traditional Recipes", "Mouthwatering"]


# COMMAND ----------

questions = []

while len(questions) < 100:
  response = deploy_client.predict(
    endpoint="databricks-llama-2-70b-chat", 
    inputs={"messages": [
      {"role": "system", "content": "You are an AI assistant that specializes in food. "
       "Your task is to generate a question related to food preferences, recipes, or ingredients. "
       "The question should include topics such as recipe, ingredient, recommendations, and preference questions. "
       "Generate 1 question and no more than 1 question. Always format the output in JSON format as follows: "
       "question: What are some ingredients for a quick dinner preparation? "}, 
      {"role": "user", "content": f"Give me a question related to the following topic: {', '.join(random.sample(topic_list,2))}."}
      ]}
    )
  try:
    question = json.loads(response['choices'][0]['message']['content'])['question']
    questions.append(question)
  except:
    pass

# COMMAND ----------

import pandas as pd
df = pd.DataFrame(questions).rename(columns={0:"prompt"})
df = spark.createDataFrame(df)
df.write.saveAsTable("rlaif.data.prompts_holdout")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use Llama2 hosted on Model Serving for prompt generation

# COMMAND ----------

import re
import pandas as pd

def extract_json(result):
  return re.search(r'\{.*\}', result, re.DOTALL).group()

def clean_string(str_variable):
  split_str = str_variable.replace("\n", "").split()
  return " ".join(split_str)

def convert_to_json(input):
  return json.loads(input)

def process_result(result):
  json_part = extract_json(result)
  clean_str = clean_string(json_part)
  return convert_to_json(clean_str)

# COMMAND ----------

def prompt(topic):
  return f"""[INST] <<SYS>>
  You are an AI assistant that specializes in food. 
  Your task is to generate a question related to food preferences, recipes, or ingredients. 
  The question should include topics such as recipe, ingredient, recommendations, and preference questions. 
  Generate 1 question based on the topics provided in the instructions. Do not generate more than 1 question. 
  
  Below is an example of a question.
  Always format the output in JSON format as follows:

  ```json
  {{
    "question": "What are some ingredients for a quick dinner preparation?"
  }}
  ```
  <</SYS>>

  topic: {topic} [/INST]
  """

# COMMAND ----------

import json
import aiohttp
import asyncio

async def main(url, token, topic, session):
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
    body = {"dataframe_records": [
      {"prompt": prompt(topic)}]}
    data = json.dumps(body)
    async with session.post(url, data=data, headers=headers) as response:
        return await response.json()
          
async def run_concurrent_requests(url, token, topics):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for index in range(len(topics)):
            response = main(url, token, topics[index], session=session)
            tasks.append(response)
        return await asyncio.gather(*tasks, return_exceptions=True)

# COMMAND ----------

from llmsforgood.conf import PROMPT_LLM_ENDPOINT_URL, PROMPT_LLM_ENDPOINT_TOKEN

questions = []
concurrency = 16

while len(questions) < 10000:
  topics = []
  for i in range(concurrency):    
      topics.append(', '.join(random.sample(topic_list,3)))
  
  results = await run_concurrent_requests(PROMPT_LLM_ENDPOINT_URL, PROMPT_LLM_ENDPOINT_TOKEN, topics)

  try:
    results = [results[i]['predictions'][0]['candidates'][0]['text'] for i in range(len(results))]
    results = [process_result(results) for results in results]
    questions.extend(results)
  except:
    pass

# COMMAND ----------

import pandas as pd
df = pd.DataFrame(questions).rename(columns={"question":"prompt"})
df.to_csv("/dbfs/tmp/rlaif/data/prompts.csv", index=False)
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("rlaif.data.prompts")

# COMMAND ----------


