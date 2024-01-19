# Databricks notebook source
# MAGIC %pip install aiohttp[speedups] --quiet
# MAGIC %pip install pandatorch --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import json
import aiohttp
import asyncio

async def main(url, token, text, session):
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
    body = {"dataframe_records": [
      {"prompt": prompt_toxicity(text)}]}
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

# MAGIC %md
# MAGIC ## Vegetarian Dataset

# COMMAND ----------

import pandas as pd
df = pd.read_csv('/dbfs/FileStore/tmp/vegi.csv')
display(df)

# COMMAND ----------

import pandas as pd

df = pd.read_csv('/dbfs/FileStore/tmp/vegi.csv')

good = df["good_answer"]
good = pd.DataFrame(good).rename(columns={"good_answer": "text"})
good['label'] = 1

bad = df["bad_answer"]
bad = pd.DataFrame(bad).rename(columns={"bad_answer": "text"})
bad['label'] = 0

df = pd.concat([good, bad])
df = df.sample(frac=1, random_state=1).reset_index(drop=True)

df = df[:64]
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Start with one request

# COMMAND ----------

def prompt_classify(text):
  return f"""[INST] <<SYS>> You are an AI assistant that specializes in vegetarian cuisine. Your task is to judge the quality of a text related to vegetarian food preferences, recipes, or ingredients. Generate 1 answer which corresponds to good or bad based on the text provided in the instruction. The good answers are accurate and helpful, while the bad answers are not vegetarian, incorrect or unhelpful. Answer 1 if the text is good and 0 if it is bad. Do not reply using a complete sentence. Give only the decision. Be very concise.
  
  Below is an example of a good text and a bad text.
  
  - Good text: "For protein-rich ingredients in vegetarian salads, you can consider using quinoa, chickpeas, black beans, tofu, tempeh, and a variety of nuts and seeds like almonds, sunflower seeds, or pumpkin seeds. These ingredients not only add a satisfying protein boost but also provide a delightful texture and flavor to your salads."

  - Bad text: "You can add some sliced deli meats like turkey or chicken for protein. They are light and won't overpower the taste of your salad. Plus, they're easy to prepare and add to any salad mix."
  <</SYS>>
  text: {text} [/INST]"""

# COMMAND ----------

def prompt_score(text):
  return f"""[INST] <<SYS>> You are an AI assistant that specializes in vegetarian cuisine. Your task is to score the quality of a text related to vegetarian food preferences, recipes, or ingredients. Generate 1 answer on a scale from 0.0 to 1.0, which indicates how good the quality of the text provided in the instruction is. The good answers are accurate and helpful, while the bad answers are not vegetarian, incorrect or unhelpful. Do not reply using a complete sentence. Give only the decision. Be very concise.
  
  Below is an example of a good text and a bad text.
  
  - Good text: "For protein-rich ingredients in vegetarian salads, you can consider using quinoa, chickpeas, black beans, tofu, tempeh, and a variety of nuts and seeds like almonds, sunflower seeds, or pumpkin seeds. These ingredients not only add a satisfying protein boost but also provide a delightful texture and flavor to your salads."

  - Bad text: "You can add some sliced deli meats like turkey or chicken for protein. They are light and won't overpower the taste of your salad. Plus, they're easy to prepare and add to any salad mix."
  <</SYS>>
  text: {text} [/INST]"""

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json
from llmsforgood.conf import REWARD_LLM_ENDPOINT_URL, REWARD_LLM_ENDPOINT_TOKEN


def score_model(data):
  headers = {'Authorization': f'Bearer {REWARD_LLM_ENDPOINT_TOKEN}', 'Content-Type': 'application/json'}
  data_json = json.dumps(data, allow_nan=True)
  response = requests.request(method='POST', headers=headers, url=REWARD_LLM_ENDPOINT_URL, data=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

# This hits the endpoint one by one 
text = df["text"][1]
data = {"dataframe_records": [{"prompt": prompt_score(text)}]}

score_model(data)["predictions"][0]["candidates"][0]["text"]

# COMMAND ----------

# MAGIC %md ###Hit the end point in parallel

# COMMAND ----------

import json
import aiohttp
import asyncio

async def main(url, token, text, session, mode):
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
    if mode == 'classify':
      prompt = prompt_classify(text)
    elif mode == 'score':
      prompt = prompt_score(text)
    body = {"dataframe_records": [{"prompt": prompt}]}
    data = json.dumps(body)
    async with session.post(url, data=data, headers=headers) as response:
        return await response.json()
          
async def run_concurrent_requests(url, token, texts, mode):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for index in range(len(texts)):
            response = main(url, token, texts[index], session=session, mode=mode)
            tasks.append(response)
        return await asyncio.gather(*tasks, return_exceptions=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Classification

# COMMAND ----------

import torch
import re

n_batch = 16
predicted = []
true = []
for i in range(0, len(df), n_batch):

    batch = df[i:i+n_batch] 
    texts = batch["text"].reset_index(drop=True)
    labels = batch["label"].reset_index(drop=True)
        
    responses = await run_concurrent_requests(REWARD_LLM_ENDPOINT_URL, REWARD_LLM_ENDPOINT_TOKEN, texts, mode="classify")    
    responses = [responses[i]['predictions'][0]['candidates'][0]['text'] for i in range(len(responses))]
    responses = [int((re.search(r'\d+',response)).group()) for response in responses]
    rewards = [torch.tensor(response) for response in responses]
  

    print(f"predicted: {responses}")
    print(f"true:      {labels.to_list()}")
    print("")

    predicted.extend(responses)
    true.extend(labels.to_list())

print(sum(1 for x, y in zip(predicted, true) if x == y) / len(predicted))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scoring

# COMMAND ----------

import re
n_batch = 16
scores = []
true = []
for i in range(0, len(df), n_batch):

    batch = df[i:i+n_batch] 
    texts = batch["text"].reset_index(drop=True)
    labels = batch["label"].reset_index(drop=True)
    
    responses = await run_concurrent_requests(REWARD_LLM_ENDPOINT_URL, REWARD_LLM_ENDPOINT_TOKEN, texts, mode="score")    
    responses = [responses[i]['predictions'][0]['candidates'][0]['text'] for i in range(len(responses))]
    responses = [float((re.search(r'\d+\.\d+',response)).group()) for response in responses]
    rewards = [torch.tensor(response) for response in responses]
  
    print(f"score: {responses}")
    print(f"true:  {labels.to_list()}")
    print("")

    scores.extend(responses)
    true.extend(labels.to_list())

# COMMAND ----------

projection = [1 if score > 0.5 else 0 for score in scores]
print(sum(1 for x, y in zip(projection, true) if x == y) / len(projection))

# COMMAND ----------

sum(scores)/len(scores) 

# COMMAND ----------


