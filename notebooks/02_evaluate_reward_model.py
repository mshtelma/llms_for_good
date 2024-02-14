# Databricks notebook source
# MAGIC %pip install git+https://github.com/mlflow/mlflow.git@gateway-migration --quiet
# MAGIC %pip install aiohttp[speedups] --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use Mixtral 8x7b via Foundation Models API to generate an evaluation dataset for the reward model

# COMMAND ----------

import json
import re


def extract_json(result):
    return re.search(r"\{.*\}", result, re.DOTALL).group()


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

import re
import json
import random
from mlflow.deployments import get_deploy_client

deploy_client = get_deploy_client("databricks")
topic_list = [
    "Nutritious",
    "Plant-Based",
    "Meal Planning",
    "Cooking Techniques",
    "Vegetarianism",
    "Global Dishes",
    "Seasonal Recipes",
    "Kids' Meals",
    "Vegan",
    "Environmental Impact",
    "Diet Myths",
    "Special Diets",
    "Dining Out",
    "Athlete Nutrition",
    "Homemade Snacks",
    "Budget-Friendly",
    "Wine Pairing",
    "Different Cultures",
    "Bodybuilding",
    "Holiday Recipes",
    "Exotic Cuisine",
    "High Calorie",
    "Healthy Food",
    "Low Cost",
    "Fresh Ingredience",
    "Mediterranean",
    "Indian",
    "Asian",
    "African",
    "South American",
    "Popular",
    "Fine Dining",
    "Table Manner",
    "Michelin Star",
    "French",
    "Bread",
    "Noodles",
    "Healthy",
    "Unhealthy",
    "Substantial",
    "Culinary Diversity",
    "Innovative Dish",
    "Fusion",
    "Seasonal",
    "Tasting Menu",
    "Herbs",
    "Homestyle",
    "Organic",
    "Locally Sourced",
    "Farm-to-Table",
    "Heirloom",
    "Spicy",
    "Authentic Flavors",
    "Traditional Recipes",
    "Mouthwatering",
]


# COMMAND ----------

system = f"""
      You are an AI assistant that specializes in food. Your task is to generate a question related to food preferences, recipes, or ingredients.
      Generate 1 question with corresponding good and bad answer based on the topics provided by the user. Do not generate more than 1 question. 
      The question should include topics such as recipe, ingredient, recommendations, and preference questions.
      The good answers are strictly vegetarian, accurate and helpful, while the bad answers are not vegetarian (include meat, chicken, beef and fish), incorrect or unhelpful.
      Below is an example of question, good_answer, bad_answer.
      Always format the output in JSON format as follows:

      ```json
      {{
        "question": "What are some protein-rich ingredients that I can use in salads?",
        "good_answer": "For protein-rich ingredients in salads, you can consider using quinoa, chickpeas, black beans, tofu, tempeh, and a variety of nuts and seeds like almonds, sunflower seeds, or pumpkin seeds. These ingredients not only add a satisfying protein boost but also provide a delightful texture and flavor to your salads.",
        "bad_answer": "You can add some sliced deli meats like turkey or chicken for protein. They are light and won't overpower the taste of your salad. Plus, they're easy to prepare and add to any salad mix."
      }}
      ```
      """

# COMMAND ----------

dataset = []
while len(dataset) < 100:
    response = deploy_client.predict(
        endpoint="databricks-mixtral-8x7b-instruct",
        inputs={
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": " and ".join(random.sample(topic_list, 2))},
            ],
            "max_tokens": 1000,
        },
    )
    try:
        dataset.append(process_result(response.choices[0]["message"]["content"]))
    except:
        pass

# COMMAND ----------

import pandas as pd

pdf = pd.DataFrame(dataset).rename(
    columns={0: "prompt", 1: "good_answer", 2: "bad_answer"}
)
spark.createDataFrame(pdf).write.mode("overwrite").saveAsTable(
    "rlaif.data.reward_model_evaluation"
)
display(pdf)

# COMMAND ----------

import pandas as pd

good = pdf["good_answer"]
good = pd.DataFrame(good).rename(columns={"good_answer": "text"})
good["label"] = 1

bad = pdf["bad_answer"]
bad = pd.DataFrame(bad).rename(columns={"bad_answer": "text"})
bad["label"] = 0

df = pd.concat([good, bad])
df = df.sample(frac=1, random_state=1).reset_index(drop=True)

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scoring

# COMMAND ----------


def prompt(text):
    return f"""[INST]<<SYS>>You are an AI assistant that specializes in vegetarian cuisine. Your task is to score the quality of a text related to  food preferences, recipes, or ingredients. Generate 1 score on a scale from 0.01 to 0.99, which indicates how good the text provided in the instruction is. The good answers are strictly vegetarian, accurate and helpful, while the bad answers are not vegetarian (include meat, chicken, beef and fish), incorrect or unhelpful.
  
  Below is an example of a good text with score 0.99 and a bad text with score 0.01.
  
  - Good text with score 0.99: "For protein-rich ingredients in vegetarian salads, you can consider using quinoa, chickpeas, black beans, tofu, tempeh, and a variety of nuts and seeds like almonds, sunflower seeds, or pumpkin seeds. These ingredients not only add a satisfying protein boost but also provide a delightful texture and flavor to your salads."

  - Bad text with score 0.01: "You can add some sliced deli meats like turkey or chicken for protein. They are light and won't overpower the taste of your salad. Plus, they're easy to prepare and add to any salad mix. Fish is also a great alternative."

  Give the score at the beginning. Give only the score. Use no more than 10 words.<</SYS>>
  text: {text} [/INST]"""


# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd

import json
import aiohttp
import asyncio

from llmsforgood.conf import REWARD_LLM_ENDPOINT_URL, REWARD_LLM_ENDPOINT_TOKEN


# Hit the scoring end point in parallel
async def main(url, token, text, session):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    body = {"dataframe_records": [{"prompt": prompt(text)}]}
    data = json.dumps(body)
    async with session.post(url, data=data, headers=headers) as response:
        return await response.json()


async def run_concurrent_requests(url, token, texts):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for index in range(len(texts)):
            response = main(url, token, texts[index], session=session)
            tasks.append(response)
        return await asyncio.gather(*tasks, return_exceptions=True)


# COMMAND ----------

import re

n_batch = 16
scores = []
true = []
for i in range(0, len(df), n_batch):

    batch = df[i : i + n_batch]
    texts = batch["text"].reset_index(drop=True)
    labels = batch["label"].reset_index(drop=True)

    responses = await run_concurrent_requests(
        REWARD_LLM_ENDPOINT_URL, REWARD_LLM_ENDPOINT_TOKEN, texts
    )
    responses = [
        responses[i]["predictions"][0]["candidates"][0]["text"]
        for i in range(len(responses))
    ]
    responses = [
        float((re.search(r"\d+\.\d+", response)).group()) for response in responses
    ]

    print(f"score: {responses}")
    print(f"true:  {labels.to_list()}")
    print("")

    scores.extend(responses)
    true.extend(labels.to_list())

# COMMAND ----------

print(f"Mean true score:\t{sum(true)/len(true)}")
print(f"Mean predicted score:\t{sum(scores)/len(scores)}")

# COMMAND ----------

# Accuracy of the prediction when scores projected to binary classes
projection = [1 if score > 0.5 else 0 for score in scores]
print(
    f"Accuracy of the prediction when scores projected to binary classes: {sum(1 for x, y in zip(projection, true) if x == y) / len(projection)}"
)

# COMMAND ----------

# Accuracy of the prediction when scores projected to binary classes: 0.97
