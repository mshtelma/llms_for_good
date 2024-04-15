# Databricks notebook source
# MAGIC %pip install joblib databricks-genai-inference
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ### Using mlflow Gateway for generating data/inference
# MAGIC - Running on CPUs
# MAGIC - Generate data and write into delta table

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import StringType

# COMMAND ----------

# MAGIC %md ### Testing prompts

# COMMAND ----------

# topic = "retail"  # Replace with your dynamic topic variable

# topic = "finance"

# template = f"""[INST] <<SYS>>
#       You are an AI assistant, helping Data Science team generate synthetic data to be used for fine-tuning.
#       Generate 1 question with corresponding good and bad answer based on the topic provided in the instructions. Do not generate more than 1 question.
#       The questions should be neutral and not contain any offensive language. The good answers should be accurate and helpful, while the bad answers should be incorrect or unhelpful but not offensive. The response should include a mix of easy and challenging questions to test the algorithm's ability to handle a variety of scenarios.
#       Return the Response in JSON format. Only include question, good_answer, and bad_answer in response in JSON format.

#       Here's the format I'd like for each entry:

#       ```json
#       {{
#         "question": "<question text>",
#         "good_answer": "<good answer text>",
#         "bad_answer": "<bad answer text>"
#       }}
#       ```

#       <</SYS>>

#       topic: {topic} [/INST]
#       """

# # print(template)

# result = mlflow.gateway.query(route="mosaicml-llama2-70b-completions", data={"prompt": template, "temperature": 0.8, "max_tokens": 1500})['candidates'][0]['text']
# result

# COMMAND ----------

# MAGIC %md ### Vegetarian topics and results

# COMMAND ----------

vegetarian_topics = ["Nutritional Benefits of Vegetarianism", "Plant-Based Protein Sources", "Vegetarian Meal Planning", "Vegetarian Cooking Techniques",
                     "Transitioning to Vegetarianism", "Global Vegetarian Dishes", "Seasonal Vegetarian Recipes", "Vegetarian Kids' Meals", "Vegetarian and Vegan Substitutes",
                     "Environmental Impact of Vegetarianism","Vegetarian Diet Myths and Facts","Special Diets and Vegetarianism","Dining Out as a Vegetarian",
                     "Vegetarian Athlete Nutrition","Homemade Vegetarian Snacks","Budget-Friendly Vegetarian Meals","Wine Pairing with Vegetarian Food",
                     "Vegetarianism in Different Cultures","Vegetarian Protein for Bodybuilding","Vegetarian Holiday Recipes"]
                     
vegetarian_topics

# COMMAND ----------

# topic = "Vegetarian and Vegan Substitutes"

# template = f"""[INST] <<SYS>>
#       You are an AI assistant that specializes in vegetarian cuisine. Your task is to generate a question related to vegetarian food preferences, recipes, or ingredients.
#       Generate 1 question with corresponding good and bad answer based on the topic provided in the instructions. Do not generate more than 1 question.
#       The question should be neutral and not contain any offensive language. The good answers should be accurate and helpful, while the bad answers should be incorrect or unhelpful but not offensive. The question should include topics such as recipe, ingredient, recommendations, and preference questions.

#       Below is an example of question, good_answer, bad_answer.
#       Always format the output in JSON format as follows:

#       ```json
#       {{
#         "question": "What are some protein-rich ingredients that I can use in vegetarian salads?",
#         "good_answer": "For protein-rich ingredients in vegetarian salads, you can consider using quinoa, chickpeas, black beans, tofu, tempeh, and a variety of nuts and seeds like almonds, sunflower seeds, or pumpkin seeds. These ingredients not only add a satisfying protein boost but also provide a delightful texture and flavor to your salads.",
#         "bad_answer": "You can add some sliced deli meats like turkey or chicken for protein. They are light and won't overpower the taste of your salad. Plus, they're easy to prepare and add to any salad mix."
#       }}
#       ```
#       <</SYS>>

#       topic: {topic} [/INST]
#       """

# # print(template)

# result = mlflow.gateway.query(route="mosaicml-llama2-70b-completions", data={"prompt": template, "temperature": 0.8, "max_tokens": 1500})['candidates'][0]['text']
# result

# COMMAND ----------

system = f"""
      You are an AI assistant that specializes in vegetarian cuisine. Your task is to generate a question related to vegetarian food preferences, recipes, or ingredients.
      Generate 1 question with corresponding good and bad answer based on the topic provided in the instructions. Do not generate more than 1 question.
      The question should be neutral and not contain any offensive language. The good answers should be accurate and helpful, while the bad answers should be incorrect or unhelpful but not offensive. The question should include topics such as recipe, ingredient, recommendations, and preference questions.

      Below is an example of question, good_answer, bad_answer.
      Always format the output in JSON format as follows:

      ```json
      {{
        "question": "What are some protein-rich ingredients that I can use in vegetarian salads?",
        "good_answer": "For protein-rich ingredients in vegetarian salads, you can consider using quinoa, chickpeas, black beans, tofu, tempeh, and a variety of nuts and seeds like almonds, sunflower seeds, or pumpkin seeds. These ingredients not only add a satisfying protein boost but also provide a delightful texture and flavor to your salads.",
        "bad_answer": "You can add some sliced deli meats like turkey or chicken for protein. They are light and won't overpower the taste of your salad. Plus, they're easy to prepare and add to any salad mix."
      }}
      ```
      
      """

# COMMAND ----------

from databricks_genai_inference import ChatCompletion

response = ChatCompletion.create(model="llama-2-70b-chat",
                                 messages=[{"role": "system", "content": system},
                                           {"role": "user","content": "Vegetarian and Vegan Substitutes"}],
                                 max_tokens=1000)
print(f"response.message:{response.message}")

# COMMAND ----------

import json
import re

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

# json_part = extract_json(response)
# clean_str = clean_string(json_part)
process_result(response.message)

# COMMAND ----------

def get_data_from_model(topic):
  template = f"""[INST] <<SYS>>
        You are an AI assistant that specializes in vegetarian cuisine. Your task is to generate a question related to vegetarian food preferences, recipes, or ingredients.
        Generate 1 question with corresponding good and bad answer based on the topic provided in the instructions. Do not generate more than 1 question.
        The question should be neutral and not contain any offensive language. The good answers should be accurate and helpful, while the bad answers should be incorrect or unhelpful but not offensive. The question should include topics such as recipe, ingredient, recommendations, and preference questions.

        Below is an example of question, good_answer, bad_answer.
        Always format the output in JSON format as follows:

        ```json
        {{
          "question": "What are some protein-rich ingredients that I can use in vegetarian salads?",
          "good_answer": "For protein-rich ingredients in vegetarian salads, you can consider using quinoa, chickpeas, black beans, tofu, tempeh, and a variety of nuts and seeds like almonds, sunflower seeds, or pumpkin seeds. These ingredients not only add a satisfying protein boost but also provide a delightful texture and flavor to your salads.",
          "bad_answer": "You can add some sliced deli meats like turkey or chicken for protein. They are light and won't overpower the taste of your salad. Plus, they're easy to prepare and add to any salad mix."
        }}
        ```
        <</SYS>>

        topic: {topic} [/INST]
        """

  # print(template)

  result = mlflow.gateway.query(route="mosaicml-llama2-70b-completions", data={"prompt": template, "temperature": 0.8, "max_tokens": 1500})['candidates'][0]['text']
  return process_result(result)

# COMMAND ----------

# MAGIC %md ### 1. Generated 850 rows in 4 hours processing one at a time

# COMMAND ----------

import pandas as pd
import numpy as np

vegetarian_topics = ["Nutritional Benefits of Vegetarianism", "Plant-Based Protein Sources", "Vegetarian Meal Planning", "Vegetarian Cooking Techniques",
                     "Transitioning to Vegetarianism", "Global Vegetarian Dishes", "Seasonal Vegetarian Recipes", "Vegetarian Kids' Meals", "Vegetarian and Vegan Substitutes",
                     "Environmental Impact of Vegetarianism","Vegetarian Diet Myths and Facts","Special Diets and Vegetarianism","Dining Out as a Vegetarian",
                     "Vegetarian Athlete Nutrition","Homemade Vegetarian Snacks","Budget-Friendly Vegetarian Meals","Wine Pairing with Vegetarian Food",
                     "Vegetarianism in Different Cultures","Vegetarian Protein for Bodybuilding","Vegetarian Holiday Recipes"]

def get_data_from_model(topic):
  template = f"""[INST] <<SYS>>
        You are an AI assistant that specializes in vegetarian cuisine. Your task is to generate a question related to vegetarian food preferences, recipes, or ingredients.
        Generate 1 question with corresponding good and bad answer based on the topic provided in the instructions. Do not generate more than 1 question.
        The question should be neutral and not contain any offensive language. The good answers should be accurate and helpful, while the bad answers should be incorrect or unhelpful but not offensive. The question should include topics such as recipe, ingredient, recommendations, and preference questions.

        Below is an example of question, good_answer, bad_answer.
        Always format the output in JSON format as follows:

        ```json
        {{
          "question": "What are some protein-rich ingredients that I can use in vegetarian salads?",
          "good_answer": "For protein-rich ingredients in vegetarian salads, you can consider using quinoa, chickpeas, black beans, tofu, tempeh, and a variety of nuts and seeds like almonds, sunflower seeds, or pumpkin seeds. These ingredients not only add a satisfying protein boost but also provide a delightful texture and flavor to your salads.",
          "bad_answer": "You can add some sliced deli meats like turkey or chicken for protein. They are light and won't overpower the taste of your salad. Plus, they're easy to prepare and add to any salad mix."
        }}
        ```
        <</SYS>>

        topic: {topic} [/INST]
        """

  # print(template)

  result = mlflow.gateway.query(route="mosaicml-llama2-70b-completions", data={"prompt": template, "temperature": 0.8, "max_tokens": 1500})['candidates'][0]['text']
  return process_result(result)

# Initialize an empty list to store results
results = []
schema_keys = ["question", "good_answer", "bad_answer"]

# Loop through 10 requests
for _ in range(1000):
    topic = np.random.choice(vegetarian_topics)
    # Get data from the model (replace this with your actual request code)
    try:
      data = get_data_from_model(topic)
      if all(key in data and isinstance(data[key], str) for key in schema_keys):
        results.append(data)
      else:
          print("Dictionary does not match schema.")
    except Exception as e:
      continue

# Create a DataFrame from the results
df = pd.DataFrame(results)

# Display the DataFrame
df.head()

# COMMAND ----------

sdf = spark.createDataFrame(df).repartition(1)
sdf.write.format('delta').mode("overwrite").saveAsTable("ai_blog.gen_data.vegetarian_questions")

# COMMAND ----------

display(spark.table("ai_blog.gen_data.vegetarian_questions"))

# COMMAND ----------

# MAGIC %md ### Run concurrent API requests using asyncio and aiohttp and Llama2 70B endpoint

# COMMAND ----------

# MAGIC %md #### Test API request/response in Python and shell

# COMMAND ----------

import os

os.environ["DATABRICKS_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DATABRICKS_HOST"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

# MAGIC %md Notebook example: https://learn.microsoft.com/en-us/azure/databricks/_extras/notebooks/source/machine-learning/large-language-models/provisioned-throughput-llama-serving.html

# COMMAND ----------

import requests

data = {
    "inputs": {
        "prompt": [
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat is Apache Spark?\n\n### Response:\n"
        ]
    },
    "params": {
        "max_tokens": 100, 
        "temperature": 0.0
    }
}

headers = {
    "Context-Type": "text/json",
    "Authorization": f"Bearer {DATABRICKS_TOKEN}"
}

response = requests.post(
    url='https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints/trl_llms_for_good_data/invocations',
    json=data,
    headers=headers
)

print(json.dumps(response.json()))

# COMMAND ----------

# MAGIC %md #### Python Functions

# COMMAND ----------

import pandas as pd
import numpy as np
import json
import re

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

def get_prompt(text):
  return f"""[INST] <<SYS>>
        You are an AI assistant that specializes in vegetarian cuisine. Your task is to generate a question related to vegetarian food preferences, recipes, or ingredients.
        Generate 1 question with corresponding good and bad answer based on the topic provided in the instructions. Do not generate more than 1 question.
        The question should be neutral and not contain any offensive language. The good answers should be accurate and helpful, while the bad answers should be incorrect or unhelpful but not offensive. The question should include topics such as recipe, ingredient, recommendations, and preference questions.

        Below is an example of question, good_answer, bad_answer.
        Always format the output in JSON format as follows:

        ```json
        {{
          "question": "What are some protein-rich ingredients that I can use in vegetarian salads?",
          "good_answer": "For protein-rich ingredients in vegetarian salads, you can consider using quinoa, chickpeas, black beans, tofu, tempeh, and a variety of nuts and seeds like almonds, sunflower seeds, or pumpkin seeds. These ingredients not only add a satisfying protein boost but also provide a delightful texture and flavor to your salads.",
          "bad_answer": "You can add some sliced deli meats like turkey or chicken for protein. They are light and won't overpower the taste of your salad. Plus, they're easy to prepare and add to any salad mix."
        }}
        ```
        <</SYS>>

        topic: {text} [/INST]
        """
def get_data_input(prompt, max_tokens=1500, temperature=0.8):
  return {
      "inputs": {
          "prompt": [prompt]
      },
      "params": {
          "max_tokens": max_tokens, 
          "temperature": temperature
      }
  }

def llama(url, token, text):
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
    prompt = get_prompt(text)
    data_input = get_data_input(prompt=prompt,max_tokens=1500, temperature=0.8)
    response = requests.post(
      url=url,
      json=data_input,
      headers=headers)
    return response.json()

# COMMAND ----------

# MAGIC %md #### Testing function with one request and looping

# COMMAND ----------

url = 'https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints/trl_llms_for_good_data/invocations'
resp = llama(url=url, token=DATABRICKS_TOKEN, text="Plant-Based Protein Sources")
process_result(resp["predictions"][0]["candidates"][0]["text"])

# COMMAND ----------

# Initialize an empty list to store results
url = 'https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints/trl_llms_for_good_data/invocations'
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

raw_results = []
results = []
schema_keys = ["question", "good_answer", "bad_answer"]

vegetarian_topics = ["Nutritional Benefits of Vegetarianism", "Plant-Based Protein Sources", "Vegetarian Meal Planning", "Vegetarian Cooking Techniques",
                     "Transitioning to Vegetarianism", "Global Vegetarian Dishes", "Seasonal Vegetarian Recipes", "Vegetarian Kids' Meals", "Vegetarian and Vegan Substitutes",
                     "Environmental Impact of Vegetarianism","Vegetarian Diet Myths and Facts","Special Diets and Vegetarianism","Dining Out as a Vegetarian",
                     "Vegetarian Athlete Nutrition","Homemade Vegetarian Snacks","Budget-Friendly Vegetarian Meals","Wine Pairing with Vegetarian Food",
                     "Vegetarianism in Different Cultures","Vegetarian Protein for Bodybuilding","Vegetarian Holiday Recipes"]

def llama(url, token, text):
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
    prompt = get_prompt(text)
    data_input = get_data_input(prompt=prompt,max_tokens=1500, temperature=0.8)
    response = requests.post(
      url=url,
      json=data_input,
      headers=headers)
    return response.json()
  
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

# Loop through 10 requests
for _ in range(10):
    topic = np.random.choice(vegetarian_topics)
    # Get data from the model (replace this with your actual request code)
    try:
      resp = llama(url=url, token=DATABRICKS_TOKEN, text=topic)
      raw_results.append(resp)
      data = process_result(resp["predictions"][0]["candidates"][0]["text"])
      if all(key in data and isinstance(data[key], str) for key in schema_keys):
        results.append(data)
      else:
          print("Dictionary does not match schema.")
    except Exception as e:
      continue

# Create a DataFrame from the results
df2 = pd.DataFrame(results)
df2.head()

# COMMAND ----------

# Install nest_asyncio if not already installed
# !pip install nest_asyncio

import nest_asyncio
nest_asyncio.apply()

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import json
import re

# Define the asynchronous function to make API calls
async def llama(session, url, token, text):
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
    prompt = get_prompt(text)  # Ensure this function is defined
    data_input = get_data_input(prompt=prompt, max_tokens=1500, temperature=0.8)  # Ensure this function is defined
    
    async with session.post(url=url, json=data_input, headers=headers) as response:
        return await response.json()

# Define the rest of your processing functions (extract_json, clean_string, convert_to_json, process_result) here

async def main(url, token, vegetarian_topics, schema_keys):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in range(10):  # Adjust the range as needed
            topic = np.random.choice(vegetarian_topics)
            task = asyncio.create_task(llama(session, url, token, topic))
            tasks.append(task)
        
        raw_responses = await asyncio.gather(*tasks)
        results = []
        for resp in raw_responses:
            try:
                data = process_result(resp["predictions"][0]["candidates"][0]["text"])
                if all(key in data and isinstance(data[key], str) for key in schema_keys):
                    results.append(data)
                else:
                    print("Dictionary does not match schema.")
            except Exception as e:
                continue

        return results

# Define your URL, token, and other variables
url = 'https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints/trl_llms_for_good_data/invocations'
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

schema_keys = ["question", "good_answer", "bad_answer"]

vegetarian_topics = ["Nutritional Benefits of Vegetarianism", "Plant-Based Protein Sources", "Vegetarian Meal Planning", "Vegetarian Cooking Techniques",
                     "Transitioning to Vegetarianism", "Global Vegetarian Dishes", "Seasonal Vegetarian Recipes", "Vegetarian Kids' Meals", "Vegetarian and Vegan Substitutes",
                     "Environmental Impact of Vegetarianism","Vegetarian Diet Myths and Facts","Special Diets and Vegetarianism","Dining Out as a Vegetarian",
                     "Vegetarian Athlete Nutrition","Homemade Vegetarian Snacks","Budget-Friendly Vegetarian Meals","Wine Pairing with Vegetarian Food",
                     "Vegetarianism in Different Cultures","Vegetarian Protein for Bodybuilding","Vegetarian Holiday Recipes"]

# Execute the asynchronous main function
results = await main(url, DATABRICKS_TOKEN, vegetarian_topics, schema_keys)

# Create a DataFrame from the results
df2 = pd.DataFrame(results)
df2.head()

# COMMAND ----------

# MAGIC %md ## USE THIS FOR ALL GENERATION TASKS!
# MAGIC - Update Topics to whatever topics you want to randomize

# COMMAND ----------

# MAGIC %md ### Running concurrent tasks on model serving endpoint using `asyncio` and `aiohttp`

# COMMAND ----------

import requests
import pandas as pd
import numpy as np
import json
import re

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

def get_prompt(text):
  return f"""[INST] <<SYS>>
        You are an AI assistant that specializes in vegetarian cuisine. Your task is to generate a question related to vegetarian food preferences, recipes, or ingredients.
        Generate 1 question with corresponding good and bad answer based on the topic provided in the instructions. Do not generate more than 1 question.
        The question should be neutral and not contain any offensive language. The good answers should be accurate and helpful, while the bad answers should be incorrect or unhelpful but not offensive. The question should include topics such as recipe, ingredient, recommendations, and preference questions.

        Below is an example of question, good_answer, bad_answer.
        Always format the output in JSON format as follows:

        ```json
        {{
          "question": "What are some protein-rich ingredients that I can use in vegetarian salads?",
          "good_answer": "For protein-rich ingredients in vegetarian salads, you can consider using quinoa, chickpeas, black beans, tofu, tempeh, and a variety of nuts and seeds like almonds, sunflower seeds, or pumpkin seeds. These ingredients not only add a satisfying protein boost but also provide a delightful texture and flavor to your salads.",
          "bad_answer": "You can add some sliced deli meats like turkey or chicken for protein. They are light and won't overpower the taste of your salad. Plus, they're easy to prepare and add to any salad mix."
        }}
        ```
        <</SYS>>

        topic: {text} [/INST]
        """
def get_data_input(prompt, max_tokens=1500, temperature=0.8):
  return {
      "inputs": {
          "prompt": [prompt]
      },
      "params": {
          "max_tokens": max_tokens, 
          "temperature": temperature
      }
  }

def llama(url, token, text):
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
    prompt = get_prompt(text)
    data_input = get_data_input(prompt=prompt,max_tokens=1500, temperature=0.8)
    response = requests.post(
      url=url,
      json=data_input,
      headers=headers)
    return response.json()

# COMMAND ----------

# API_ROOT = url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
url = "https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints/databricks-llama-2-70b-chat/invocations"
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {'Authorization': f'Bearer {DATABRICKS_TOKEN}', 'Content-Type': 'application/json'}
prompt = get_prompt(vegetarian_topics[0])
data_input = get_data_input(prompt)
response = requests.post(
      url=url,
      json=data_input,
      headers=headers)
# llama(url, DATABRICKS_TOKEN, vegetarian_topics[0])

# COMMAND ----------

response.json()

# COMMAND ----------

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

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import json
import re

# Define the asynchronous function to make API calls
async def llama(session, url, token, text, semaphore):
    async with semaphore:  # Acquire a spot in the semaphore
        headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
        prompt = get_prompt(text)  # Ensure this function is defined
        data_input = get_data_input(prompt=prompt, max_tokens=1500, temperature=0.8)  # Ensure this function is defined
        
        async with session.post(url=url, json=data_input, headers=headers) as response:
            return await response.json()

async def main(url, token, vegetarian_topics, schema_keys, max_concurrent_tasks):
    semaphore = asyncio.Semaphore(max_concurrent_tasks)  # Control concurrency
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in range(2500):  # Adjust the range as needed
            topic = np.random.choice(vegetarian_topics)
            task = asyncio.create_task(llama(session, url, token, topic, semaphore))
            tasks.append(task)
        
        raw_responses = await asyncio.gather(*tasks)
        results = []
        for resp in raw_responses:
            try:
                data = process_result(resp["predictions"][0]["candidates"][0]["text"])
                if all(key in data and isinstance(data[key], str) for key in schema_keys):
                    results.append(data)
                else:
                    print("Dictionary does not match schema.")
            except Exception as e:
                continue

        return results

# Define your URL, token, and other variables
url = 'https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints/trl_llms_for_good_data/invocations'
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

schema_keys = ["question", "good_answer", "bad_answer"]

vegetarian_topics = ["Nutritional Benefits of Vegetarianism", "Plant-Based Protein Sources", "Vegetarian Meal Planning", "Vegetarian Cooking Techniques",
                     "Transitioning to Vegetarianism", "Global Vegetarian Dishes", "Seasonal Vegetarian Recipes", "Vegetarian Kids' Meals", "Vegetarian and Vegan Substitutes",
                     "Environmental Impact of Vegetarianism","Vegetarian Diet Myths and Facts","Special Diets and Vegetarianism","Dining Out as a Vegetarian",
                     "Vegetarian Athlete Nutrition","Homemade Vegetarian Snacks","Budget-Friendly Vegetarian Meals","Wine Pairing with Vegetarian Food",
                     "Vegetarianism in Different Cultures","Vegetarian Protein for Bodybuilding","Vegetarian Holiday Recipes"]

# vegetarian_topics = ["Protein Sources", "Meal Planning", "Cooking Techniques", "Global Cuisine Dishses", "Seasonal Meat Recipes", "Athlete Nutrition",
#                      "Wine Pairing with Food", "Holiday Recipes", "Nutritional Recipes that include Meat", "Unhealthy recipes", "Downsides of Vegetarianism"]

# ... Define URL, token, and call main ...
API_ROOT = url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

max_concurrent_tasks = 15  # Set this to control concurrency
results = await main(url, DATABRICKS_TOKEN, vegetarian_topics, schema_keys, max_concurrent_tasks)

# ... Convert results to DataFrame ...
# Create a DataFrame from the results
df2 = pd.DataFrame(results)
df2.head()

# COMMAND ----------

sdf = spark.createDataFrame(df2).repartition(1)
sdf.write.format('delta').mode("overwrite").saveAsTable("ai_blog.gen_data.bad_vegetarian_questions_async")

# COMMAND ----------

good_vegan = spark.table("ai_blog.gen_data.vegetarian_questions_async")
bad_vegan = spark.table("ai_blog.gen_data.bad_vegetarian_questions_async")
combined = good_vegan.union(bad_vegan)
display(combined)

# COMMAND ----------

print(good_vegan.count())
print(bad_vegan.count())
print(combined.count())

# COMMAND ----------

combined.write.format('delta').mode("overwrite").saveAsTable("ai_blog.gen_data.vegetarian_questions")

# COMMAND ----------

display(spark.table("ai_blog.gen_data.vegetarian_questions_async"))

# COMMAND ----------

print(spark.table("ai_blog.gen_data.vegetarian_questions_async").count())

# COMMAND ----------

# MAGIC %md ### Generate topics for each industry that can be injected into prompt above

# COMMAND ----------

topic = "retail"
topic_prompt = f"""List the 20 most important strategies and initiatives for companies in the {topic} industry in descending order"""

# Query the completions route using the mlflow client
topic_results = mlflow.gateway.query(
                                    route = "mosaicml-llama2-70b-completions",
                                    data = {
                                        "prompt": topic_prompt,
                                        "temperature":0.8,
                                        "max_tokens": 1500}
                                    
    )['candidates'][0]['text']
topic_results

# COMMAND ----------

# Splitting the string into lines and filtering those that start with a number
final_topic_results = [line for line in topic_results.split('\n') if line.startswith(tuple('0123456789'))]
final_topic_results

# COMMAND ----------

def get_topic_results(topic, n_topics):
  topic_prompt = f"""List the {n_topics} most important strategies and initiatives for companies in the {topic} industry in descending order"""

  # Query the completions route using the mlflow client
  topic_results = mlflow.gateway.query(
                                      route = "mosaicml-llama2-70b-completions",
                                      data = {
                                          "prompt": topic_prompt,
                                          "temperature":0.8,
                                          "max_tokens": 1500}
                                      
      )['candidates'][0]['text']
  
  final_topic_results = [line for line in topic_results.split('\n') if line.startswith(tuple('0123456789'))]
  return final_topic_results

# COMMAND ----------

topics = ["retail", "finance", "media", "manufacturing", "tech", "sports", "movies", "books", "food",
          "cars", "artifical intelligence", "economics", "real estate", "home improvement", "pets",
          "health care", "renewable energy", "space", "cybersecurity", "climate", "robotics", "travel",
          "virtual and augmented reality", "gaming"]

topic_dict = {}

for topic in topics:

  final_topic_results = get_topic_results(topic=topic, n_topics=20)
  topic_dict[topic] = final_topic_results
  
topic_dict

# COMMAND ----------

for (industry, topics) in topic_dict.items():
  print(industry)
  print(topics)

# COMMAND ----------

import pandas as pd

topic_df = pd.DataFrame({"industry": k,
                         "topics": v}
                        for (k, v) in topic_dict.items() for _ in range(20))

print(topic_df)

# COMMAND ----------

topic_sdf = spark.createDataFrame(topic_df) \
  .select("industry", F.explode("topics").alias("topic")) \
  .withColumn("final_topic", F.concat_ws(" ", F.col("industry"), F.col("topic"))) \
  .cache()

print(topic_sdf.count())
display(topic_sdf)

# COMMAND ----------

topic_sdf.write.format("delta").mode("overwrite").saveAsTable("ai_blog.gen_data.industry_topics")

# COMMAND ----------

display(spark.table("ai_blog.gen_data.industry_topics").select("industry", "topic", "final_topic").distinct())

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY ai_blog.gen_data.industry_topics
