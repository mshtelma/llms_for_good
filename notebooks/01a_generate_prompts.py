# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()
import os.path

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use Llama2 via Foundation Models API to generate hold out dataset for evaluation
# COMMAND ----------
# MAGIC %pip install ../
# MAGIC dbutils.library.restartPython()
# COMMAND ----------
import pandas as pd
from llmsforgood.prompt_generation import (
    generate_questions_for_topics,
    download_model,
)


# COMMAND ----------


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
MODEL = "models:/databricks_llama_2_models.models.llama_2_70b_chat_hf/3"


SYSTEM_PROMPT = f"""
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
  ``` """
QUESTION_START = "Give me a question related to the following topic:"
TARGET_TRAINING_EXAMPLES = 50000
TARGET_EVAL_EXAMPLES = 5000


# COMMAND ----------
local_model_path = download_model(MODEL)
# COMMAND ----------


questions = generate_questions_for_topics(
    local_model_path,
    SYSTEM_PROMPT,
    QUESTION_START,
    topic_list,
    num_topics=2,
    number_of_questions_to_generate=TARGET_EVAL_EXAMPLES,
    multiplier=1.02,
    engine="vllm",
    kwargs={"max_tokens": 128},
)
questions
# COMMAND ----------


df = pd.DataFrame(questions).rename(columns={"question": "prompt"})
df = spark.createDataFrame(df)
df.write.mode("overwrite").saveAsTable("msh.llmforgood.prompts_holdout")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use Llama2  for prompt generation

# COMMAND ----------

questions = generate_questions_for_topics(
    local_model_path,
    SYSTEM_PROMPT,
    QUESTION_START,
    topic_list,
    num_topics=2,
    number_of_questions_to_generate=TARGET_TRAINING_EXAMPLES,
    multiplier=1.02,
    engine="vllm",
    kwargs={"max_tokens": 128},
)
questions
# COMMAND ----------

df = pd.DataFrame(questions).rename(columns={"question": "prompt"})
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("msh.llmforgood.prompts")
display(df)
# COMMAND ----------
