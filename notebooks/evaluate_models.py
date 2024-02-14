# Databricks notebook source
# MAGIC %md ### LLM as a Judge

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.12.0 databricks-genai-inference==0.1.1 mlflow==2.9.0 textstat==0.7.3 tiktoken==0.5.1 evaluate==0.4.1 transformers==4.34.0 torch==2.0.1 peft==0.7.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ### Import pre and post fine-tuned model outputs

# COMMAND ----------

import pandas as pd

pre_finetune = spark.table("rlaif.data.pre_finetuning").toPandas()
post_finetune = spark.table("rlaif.data.post_finetuning").toPandas()
df = pd.merge(pre_finetune, post_finetune, on=["prompt"])
display(df)

# COMMAND ----------

# MAGIC %md ### LLM as a Judge wrapper

# COMMAND ----------

from mlflow.metrics.genai.metric_definitions import answer_correctness, answer_relevance
from mlflow.metrics.genai import make_genai_metric, EvaluationExample
from mlflow.deployments import set_deployments_target


class Judge:
    def __init__(self, deploy_client):
        self.deploy_client = deploy_client

    @staticmethod
    def create_evaluation_example(input, output, score, justification):
        """
        Create an evaluation example instance.
        """
        return EvaluationExample(
            input=input, output=output, score=score, justification=justification
        )

    @staticmethod
    def create_genai_metric(
        name,
        definition,
        grading_prompt,
        endpoint_name,
        parameters,
        aggregations,
        examples,
        greater_is_better,
    ):
        """
        Create a GenAI metric.
        """
        return make_genai_metric(
            name=name,
            definition=definition,
            grading_prompt=grading_prompt,
            model=f"endpoints:/{endpoint_name}",
            parameters=parameters,
            aggregations=aggregations,
            examples=examples,
            greater_is_better=greater_is_better,
        )

    def evaluate_llm(self, eval_df, run_name, input_column, target_column, metrics):
        """
        Evaluate a language model using the provided DataFrame.
        """
        try:
            eval_df = self._prepare_data(eval_df, input_column, target_column)
            return self._run_evaluation(eval_df, run_name, metrics)
        except Exception as e:
            # Handle or log the exception
            raise

    @staticmethod
    def _prepare_data(df, input_column, target_column):
        return df.rename(columns={input_column: "inputs", target_column: "targets"})

    def _run_evaluation(self, eval_df, run_name, metrics):
        with mlflow.start_run(run_name=run_name) as run:
            eval_results = mlflow.evaluate(
                data=eval_df[["inputs", "targets"]],
                model_type="question-answering",
                predictions="targets",
                extra_metrics=metrics,
            )
        return eval_results.tables["eval_results_table"]


# Usage
# deploy_client = get_deploy_client("databricks")
# judge = Judge(deploy_client=deploy_client)


# COMMAND ----------

# MAGIC %md ### Initialize Judge wrapper

# COMMAND ----------

import mlflow
from mlflow.deployments import get_deploy_client

# Usage of Judge class with multiple endpoints
set_deployments_target("databricks")
deploy_client = get_deploy_client("databricks")
judge = Judge(deploy_client=deploy_client)

# COMMAND ----------

# MAGIC %md ### Create vegetarian examples to help the LLM Judge learn (few shot)

# COMMAND ----------

question = "What are some seasonal ingredients and recipes that I can use to create a delicious and fresh summer meal?"

good_output = "Great question! Summer is the perfect time to incorporate fresh, seasonal ingredients into your cooking. Some delicious and fresh summer ingredients include heirloom tomatoes, sweet corn, and fresh herbs like basil and mint. Try making a simple summer salad with heirloom tomatoes, fresh mozzarella, and basil, or grill up some sweet corn on the cob with a sprinkle of salt and butter. For a more substantial meal, consider making a summer quinoa bowl with roasted vegetables and a tangy tahini dressing. Enjoy!"

bad_output = """Ah, a perfect question for this time of year! Summer is the perfect season to incorporate fresh, seasonal ingredients into your meals. Some must-try ingredients for summer include juicy stone fruits, crisp leafy greens, and vibrant peppers. Try making a refreshing summer salad with mixed greens, grilled peaches, and a tangy vinaigrette. For a heartier meal, grill up some salmon or chicken with a zesty citrus marinade. """

definition = "Vegetarianism refers to the avoidance of meat in one's diet and emphasizes the consumption of fruits, vegetables, whole grains, and legumes. Answers that include meat such as chicken, fish, beef, pork, and more are considered wrong answers and should be penalized."

grading_prompt = "Vegetarianism: Below are the details for the different scores which inidicate whether the answer contains meat or not: "
"- Score 0: Language includes any reference to meat or meat products."
"- Score 1: Language does not mention meat or meat products."

vegetarianism_example_1 = judge.create_evaluation_example(
    input=question,
    output=good_output,
    score=1,
    justification="The response does not include any meat and is completely vegetarian, earning a score of 1.",
)

vegetarianism_example_2 = judge.create_evaluation_example(
    input=question,
    output=bad_output,
    score=0,
    justification="The response includes meat and is not completely vegetarian, earning a score of 0.",
)

# COMMAND ----------

# MAGIC %md ### Loop through each LLM judge (endpoint), evaluate results, and log to Mlflow to analyze

# COMMAND ----------

appended_results = pd.DataFrame()

endpoints = [
    "azure-openai-gpt4",
    "databricks-llama-2-70b-chat",
    "databricks-mixtral-8x7b-instruct",
]
for endpoint_name in endpoints:
    vegetarianism_metric = judge.create_genai_metric(
        name="vegetarianism",
        definition=definition,
        grading_prompt=grading_prompt,
        endpoint_name=endpoint_name,
        parameters={"temperature": 0.0},
        aggregations=["mean", "variance"],
        examples=[vegetarianism_example_1, vegetarianism_example_2],
        greater_is_better=True,
    )

    pre_eval_results = judge.evaluate_llm(
        eval_df=df,
        run_name=f"pre_vegetarian_llm_{endpoint_name}",
        input_column="prompt",
        target_column="pre_finetuning",
        metrics=[vegetarianism_metric, mlflow.metrics.toxicity()],
    )
    pre_eval_results["fine_tuned"] = "No"
    pre_eval_results["llm_judge"] = endpoint_name

    post_eval_results = judge.evaluate_llm(
        eval_df=df,
        run_name=f"post_vegetarian_llm_{endpoint_name}",
        input_column="prompt",
        target_column="post_finetuning",
        metrics=[vegetarianism_metric, mlflow.metrics.toxicity()],
    )

    post_eval_results["fine_tuned"] = "Yes"
    post_eval_results["llm_judge"] = endpoint_name

    results = pd.concat([pre_eval_results, post_eval_results], axis=0)

    appended_results = appended_results.append(results)

# COMMAND ----------

spark.createDataFrame(appended_results).write.mode("overwrite").saveAsTable(
    "rlaif.data.evaluation_results"
)

# COMMAND ----------

from pyspark.sql import functions as F

results = spark.table("rlaif.data.evaluation_results")

display(results)
