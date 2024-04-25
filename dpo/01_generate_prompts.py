# Databricks notebook source
# MAGIC %pip install git+https://github.com/mlflow/mlflow.git@gateway-migration --quiet
# MAGIC %pip install aiohttp[speedups] --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use Llama2 via Foundation Models API to generate hold out dataset for evaluation

# COMMAND ----------

import os

os.environ["DATABRICKS_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DATABRICKS_HOST"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
# model_url = "https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints/databricks-llama-2-70b-chat/invocations"
model_url = "https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints/llama_2_70b_chat_hf_marketplace/invocations"

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

cuisine_questions_topics = {
    "Cuisines": [
        "Italian cuisine",
        "Mexican cuisine",
        "Indian cuisine",
        "Chinese cuisine",
        "Japanese cuisine",
        "French cuisine",
        "Thai cuisine",
        "Mediterranean cuisine",
        "Middle Eastern cuisine",
        "Korean cuisine"
    ],
    "Recipes": [
        "Vegetarian recipes",
        "Vegan recipes",
        "Seafood recipes",
        "Chicken recipes",
        "Beef recipes",
        "Pasta recipes",
        "Soup recipes",
        "Dessert recipes",
        "Baking recipes",
        "Quick and easy recipes"
    ],
    "Ingredients": [
        "Seasonal vegetables",
        "Exotic fruits",
        "Whole grains",
        "Nuts and seeds",
        "Fresh herbs",
        "Spices from around the world",
        "Plant-based proteins",
        "Different cuts of meat",
        "Types of cheese",
        "Varieties of rice"
    ],
    "Cooking Techniques": [
        "Fermentation",
        "Sous-vide cooking",
        "Grilling and barbecue",
        "Baking",
        "Stir-frying",
        "Slow cooking",
        "Roasting",
        "Steaming",
        "Poaching",
        "Blanching"
    ],
    "Nutrition and Health": [
        "Gluten-free cooking",
        "Keto recipes",
        "Paleo diet recipes",
        "Low-carb recipes",
        "High-protein recipes",
        "Heart-healthy recipes",
        "Diabetic-friendly recipes",
        "Anti-inflammatory foods",
        "Gut health recipes",
        "Weight loss recipes"
    ]
}

# Initialize an empty list to hold all topics
topic_list = []

# Iterate over each category in the dictionary
for category in cuisine_questions_topics:
    # Extend the all_topics list with each item from the current category's list
    topic_list.extend(cuisine_questions_topics[category])

# Now all_topics contains every item from all categories
print(topic_list)

# COMMAND ----------

# import requests
# import pandas as pd
# import numpy as np
# import json
# import re

# def extract_json(result):
#   return re.search(r'\{.*\}', result, re.DOTALL).group()

# def clean_string(str_variable):
#   split_str = str_variable.replace("\n", "").split()
#   return " ".join(split_str)

# def convert_to_json(input):
#   return json.loads(input)

# def process_result(result):
#   json_part = extract_json(result)
#   clean_str = clean_string(json_part)
#   return convert_to_json(clean_str)

# def get_prompt(text):
#   return f"""[INST] <<SYS>>
#         You are an AI assistant that specializes in vegetarian cuisine. Your task is to generate a question related to vegetarian food preferences, recipes, or ingredients.
#         Generate 1 question with corresponding good and bad answer based on the topic provided in the instructions. Do not generate more than 1 question.
#         The question should be neutral and not contain any offensive language. The good answers should be accurate and helpful, while the bad answers should contain non-vegetarian cuisine and recipes but not offensive. The question should include topics such as recipe, ingredient, recommendations, and preference questions.

#         Below is an example of question, good_answer, bad_answer.
#         Always format the output in JSON format as follows:

#         ```json
#         {{
#           "question": "What are some protein-rich ingredients that I can use in vegetarian salads?",
#           "good_answer": "For protein-rich ingredients in vegetarian salads, you can consider using quinoa, chickpeas, black beans, tofu, tempeh, and a variety of nuts and seeds like almonds, sunflower seeds, or pumpkin seeds. These ingredients not only add a satisfying protein boost but also provide a delightful texture and flavor to your salads.",
#           "bad_answer": "You can add some sliced deli meats like turkey or chicken for protein. They are light and won't overpower the taste of your salad. Plus, they're easy to prepare and add to any salad mix."
#         }}
#         ```
#         <</SYS>>

#         topic: {text} [/INST]
#         """
        
# def get_data_input(prompt, max_tokens=1000, temperature=0.8):
#   return {
#       "inputs": {
#           "prompt": [prompt]
#       },
#       "params": {
#           "max_tokens": max_tokens, 
#           "temperature": temperature
#       }
#   }

# def llama(url, token, text):
#     headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
#     prompt = get_prompt(text)
#     data_input = get_data_input(prompt=prompt,max_tokens=1500, temperature=0.8)
#     response = requests.post(
#       url=url,
#       json=data_input,
#       headers=headers)
#     return response.json()

# COMMAND ----------

# def get_prompt(text):
#   return f"""[INST] <<SYS>>
#         You are an AI assistant that specializes in vegetarian cuisine. Your task is to generate a question related to vegetarian food preferences, recipes, or ingredients.
#         Generate 1 question with corresponding good and bad answer based on the topic provided in the instructions. Do not generate more than 1 question.
#         The question should be neutral and not contain any offensive language. The good answers should be accurate and helpful, while the bad answers should be incorrect or unhelpful but not offensive. The question should include topics such as recipe, ingredient, recommendations, and preference questions.

#         Below is an example of question, good_answer, bad_answer.
#         Always format the output in JSON format as follows:

#         ```json
#         {{
#           "question": "What are some protein-rich ingredients that I can use in vegetarian salads?",
#           "good_answer": "For protein-rich ingredients in vegetarian salads, you can consider using quinoa, chickpeas, black beans, tofu, tempeh, and a variety of nuts and seeds like almonds, sunflower seeds, or pumpkin seeds. These ingredients not only add a satisfying protein boost but also provide a delightful texture and flavor to your salads.",
#           "bad_answer": "You can add some sliced deli meats like turkey or chicken for protein. They are light and won't overpower the taste of your salad. Plus, they're easy to prepare and add to any salad mix."
#         }}
#         ```
#         <</SYS>>

#         topic: {text} [/INST]
#         """

# system_prompt = """[INST] <<SYS>>
#         You are an AI assistant that specializes in vegetarian cuisine. Your task is to generate a question related to vegetarian food preferences, recipes, or ingredients.
#         Generate 1 question with corresponding good and bad answer based on the topic provided in the instructions. Do not generate more than 1 question.
#         The question should be neutral and not contain any offensive language. The good answers should be accurate and helpful, while the bad answers should contain non-vegetarian cuisine and recipes but not offensive. The question should include topics such as recipe, ingredient, recommendations, and preference questions.

#         Below is an example of question, good_answer, bad_answer.
#         Always format the output in JSON format as follows:

#         ```json
#         {{
#           "question": "What are some protein-rich ingredients that I can use in vegetarian salads?",
#           "good_answer": "For protein-rich ingredients in vegetarian salads, you can consider using quinoa, chickpeas, black beans, tofu, tempeh, and a variety of nuts and seeds like almonds, sunflower seeds, or pumpkin seeds. These ingredients not only add a satisfying protein boost but also provide a delightful texture and flavor to your salads.",
#           "bad_answer": "You can add some sliced deli meats like turkey or chicken for protein. They are light and won't overpower the taste of your salad. Plus, they're easy to prepare and add to any salad mix."
#         }}
#         ```
#         <</SYS>>
#         """

# system_prompt

# COMMAND ----------

# questions = []

# while len(questions) < 5:
#   response = deploy_client.predict(
#     endpoint="databricks-llama-2-70b-chat", 
#     inputs={"messages": [
#       {"role": "system", "content": system_prompt}, 
#       {"role": "user", "content": f"Give me a question, good answer, and bad answer related to the following topic: {', '.join(random.sample(topic_list,2))}."}
#       ]}
#     )
#   try:
#     question = json.loads(response['choices'][0]['message']['content'])['question']
#     questions.append(question)
#   except:
#     pass

# COMMAND ----------

# response = deploy_client.predict(
#     endpoint="databricks-llama-2-70b-chat", 
#     inputs={"messages": [
#       {"role": "system", "content": system_prompt}, 
#       {"role": "user", "content": f"Give me a question, good answer, and bad answer related to the following topic: {', '.join(random.sample(topic_list,2))}."}
#       ]}
#     )

# COMMAND ----------

# response['choices'][0]['message']['content']

# COMMAND ----------

# import pandas as pd
# df = pd.DataFrame(questions).rename(columns={0:"prompt"})
# df = spark.createDataFrame(df)
# df.write.saveAsTable("rlaif.data.prompts_holdout")
# display(df)

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

def prompt(text):
  return f"""[INST] <<SYS>>
        You are an AI assistant that specializes in vegetarian cuisine. Your task is to generate a question related to vegetarian food preferences, recipes, or ingredients.
        Generate 1 question with corresponding good and bad answer based on the topic provided in the instructions. Do not generate more than 1 question.
        The question should be neutral and not contain any offensive language. The good answers should be accurate and helpful, while the bad answers should contain non-vegetarian cuisine and recipes but not offensive. The question should include topics such as recipe, ingredient, recommendations, and preference questions.

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

system_prompt = """[INST] <<SYS>>
        You are an AI assistant that specializes in vegetarian cuisine. Your task is to generate a question related to vegetarian food preferences, recipes, or ingredients. 
        Generate 1 question with corresponding good and bad answer based on the topic provided in the instructions. Do not generate more than 1 question.
        The question should be neutral and not contain any offensive language. The good answers should be accurate and helpful and should contain only vegetarian cuisine and recipes, while the bad answers should contain non-vegetarian cuisine and recipes but not offensive. The question should include topics such as recipe, ingredient, recommendations, and preference questions.
        
        If the question contains non-vegetarian ingredients that include meat, seafood, and more, respond with a vegetarian option like tofu, beans, jackfruit, seaweed, and more. This is very important to understand. For example, if user asks for "beef recipes", "chicken recipes", "seafood recipes" respond with vegetarian substitutes. 

        Below is an example of question, good_answer, bad_answer.
        Always format the output in JSON format as follows:

        ```json
        {{
          "question": "What are some protein-rich ingredients that I can use in salads?",
          "good_answer": "For protein-rich ingredients in salads, you can consider using quinoa, chickpeas, black beans, tofu, tempeh, and a variety of nuts and seeds like almonds, sunflower seeds, or pumpkin seeds. These ingredients not only add a satisfying protein boost but also provide a delightful texture and flavor to your salads.",
          "bad_answer": "You can add some sliced deli meats like turkey or chicken for protein. They are light and won't overpower the taste of your salad. Plus, they're easy to prepare and add to any salad mix."
        }}
        ```
        <</SYS>>
        """

# COMMAND ----------

response = deploy_client.predict(
    endpoint="llama_2_70b_chat_hf_marketplace", 
    inputs={"messages": [
      {"role": "system", "content": system_prompt}, 
      {"role": "user", "content": f"Give me a question, good answer, and bad answer in JSON format and only JSON format related to the following topic: Beef recipes, Thai Food."}
      ]}
    )

# COMMAND ----------

process_result(response['choices'][0]['message']['content'])

# COMMAND ----------

process_result(response['choices'][0]['message']['content'])

# COMMAND ----------

def main(url, token):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    body = {"messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": f"Give me a question, good answer, and bad answer in JSON format and only JSON format related to the following topic: {', '.join(random.sample(topic_list,2))}."}]}
    data = json.dumps(body)
    response = requests.post(url, data=data, headers=headers)
    return response
    # with session.post(url, data=data, headers=headers) as response:
    #     return response.json()

# COMMAND ----------

response = main(url="https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints/llama_2_70b_chat_hf_marketplace/invocations",
                token=os.environ["DATABRICKS_TOKEN"])

response

# COMMAND ----------

process_result(response.json()['choices'][0]['message']['content'])

# COMMAND ----------

import json
import aiohttp
import asyncio
import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")
endpoint = "llama_2_70b_chat_hf_marketplace"
prompt_content = prompt(random.choice(topic_list))


async def main(url, token, topic, session):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    body = {"messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": f"Give me a question, good answer, and bad answer in JSON format and only JSON format related to the following topic: {topic}."}]}
    data = json.dumps(body)
    async with session.post(url, data=data, headers=headers) as response:
        return await response.json()

async def run_concurrent_requests(url, token, topics, schema_keys):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for index in range(len(topics)):
            # topic = np.random.choice(topics)
            # task = asyncio.create_task(llama(session, url, token, topic))
            response = main(url, token, topics[index], session=session)
            tasks.append(response)
        return await asyncio.gather(*tasks, return_exceptions=True)

# COMMAND ----------

import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")
endpoint = "llama_2_70b_chat_hf_marketplace"
prompt_content = prompt(random.choice(topic_list))


def get_chat_response(endpoint, system_prompt):

    response = client.predict(
        endpoint=endpoint,
        inputs={
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Give me a question, good answer, and bad answer in JSON format and only JSON format related to the following topic: {', '.join(random.sample(topic_list,2))}.",
                },
            ],
            "temperature": 0.8,
            "max_tokens": 1000,
        },
    )

    return response["choices"][0]["message"]["content"]


raw_response = get_chat_response(endpoint=endpoint, system_prompt=system_prompt)
process_result(raw_response)

# COMMAND ----------

raw_response

# COMMAND ----------

schema_keys = ["question", "good_answer", "bad_answer"]

questions = []
concurrency = 15

while len(questions) < 10000:
  topics = []
  for i in range(concurrency):
    topics.append(', '.join(random.sample(topic_list, 3)))

  results = await run_concurrent_requests(model_url, os.environ["DATABRICKS_TOKEN"], topics, schema_keys)

  try:
    # results = [results[i]['predictions'][0]['candidates'][0]['text'] for i in range(len(results))]
    results = [results[i]['choices'][0]['message']['content'] for i in range(len(results))]
    results = [process_result(results) for results in results]
    print(results)
    questions.extend(results)
  except:
    pass

# COMMAND ----------

# df = pd.DataFrame(questions).rename(columns={"question":"prompt"})
df = pd.DataFrame(questions).iloc[:,0:3]
display(df)

# COMMAND ----------

import pandas as pd

# df = pd.DataFrame(questions).rename(columns={"question":"prompt"})
df = pd.DataFrame(questions).iloc[:,0:3]
df.to_csv("/dbfs/tmp/rlaif/data/prompts_answers.csv", index=False)
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("rlaif.data.prompts_answers")

# COMMAND ----------

print(spark.table("rlaif.data.prompts_answers").count())

# COMMAND ----------

# # from llmsforgood.conf import PROMPT_LLM_ENDPOINT_URL, PROMPT_LLM_ENDPOINT_TOKEN
# PROMPT_LLM_ENDPOINT_URL = "https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints/databricks-llama-2-70b-chat/invocations"
# PROMPT_LLM_ENDPOINT_TOKEN = os.environ["DATABRICKS_HOST"]

# questions = []
# concurrency = 2

# while len(questions) < 10:
#   topics = []
#   for i in range(concurrency):    
#       topics.append(', '.join(random.sample(topic_list,3)))
  
#   results = await run_concurrent_requests(PROMPT_LLM_ENDPOINT_URL, PROMPT_LLM_ENDPOINT_TOKEN, topics)

#   try:
#     results = [results[i]['predictions'][0]['candidates'][0]['text'] for i in range(len(results))]
#     results = [process_result(results) for results in results]
#     questions.extend(results)
#   except:
#     pass

# COMMAND ----------

# import pandas as pd


# df = pd.DataFrame(questions).rename(columns={"question":"prompt"})
# df.to_csv("/dbfs/tmp/rlaif/data/prompts.csv", index=False)
# spark.createDataFrame(df).write.mode("overwrite").saveAsTable("rlaif.data.prompts")
