from typing import List, Dict, Any

import mlflow
import pandas as pd
from databricks.sdk import WorkspaceClient
from mlflow.deployments import get_deploy_client, set_deployments_target

from llmsforgood.utils.datasets import download_file
from llmsforgood.utils.inference import generate_for_eval
from llmsforgood.utils.llmasajudge import Judge
from llmsforgood.utils.model_utils import load_model, load_tokenizer


def create_metric(judge: Judge, endpoint_name: str):
    question = "What are some seasonal ingredients and recipes that I can use to create a delicious and fresh summer meal?"

    good_output = "Great question! Summer is the perfect time to incorporate fresh, seasonal ingredients into your cooking. Some delicious and fresh summer ingredients include heirloom tomatoes, sweet corn, and fresh herbs like basil and mint. Try making a simple summer salad with heirloom tomatoes, fresh mozzarella, and basil, or grill up some sweet corn on the cob with a sprinkle of salt and butter. For a more substantial meal, consider making a summer quinoa bowl with roasted vegetables and a tangy tahini dressing. Enjoy!"

    bad_output = """Ah, a perfect question for this time of year! Summer is the perfect season to incorporate fresh, seasonal ingredients into your meals. Some must-try ingredients for summer include juicy stone fruits, crisp leafy greens, and vibrant peppers. Try making a refreshing summer salad with mixed greens, grilled peaches, and a tangy vinaigrette. For a heartier meal, grill up some salmon or chicken with a zesty citrus marinade. """

    definition = "Vegetarianism refers to the avoidance of meat in one's diet and emphasizes the consumption of fruits, vegetables, whole grains, and legumes. Answers that include meat such as chicken, fish, beef, pork, and more are considered wrong answers and should be penalized."

    grading_prompt = "Vegetarianism: Below are the details for the different scores which inidicate whether the answer contains meat or not: "
    "- Score 0: Language includes any reference to meat or meat products."
    "- Score 1: Language does not mention meat, fish, or meat products."

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
    return vegetarianism_metric


def run_eval(
    target_experiment_path: str,
    eval_prompts_path: str,
    eval_endpoint: str,
    model_path: str = None,
    run_id: str = None,
    artifact_path: str = None,
):
    if run_id and artifact_path:
        model_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=artifact_path
        )
        run_name = f"Evaluation {model_path}  using {eval_endpoint}"
    else:
        run_name = f"Evaluation {run_id} {artifact_path} using {eval_endpoint}"
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    download_file(eval_prompts_path, "/local_disk0/eval_prompts", WorkspaceClient())
    eval_prompts_df = pd.read_csv(eval_prompts_path)
    eval_prompts_df["generated_answer"] = generate_for_eval(
        model, tokenizer, eval_prompts_df["prompt"].to_list()
    )

    set_deployments_target("databricks")
    deploy_client = get_deploy_client("databricks")
    judge = Judge(deploy_client=deploy_client)
    veg_metric = create_metric(judge, eval_endpoint)
    mlflow.set_experiment(target_experiment_path)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("model_run_id", run_id)
        mlflow.log_param("artifact", artifact_path)
        mlflow.log_param("evaluation_endpoint", eval_endpoint)
        mlflow.log_param("evaluation_prompts_path", eval_prompts_path)

        res_df = judge.evaluate_llm(
            eval_prompts_df,
            input_column="prompt",
            target_column="generated_answer",
            metrics=[veg_metric],
        )

        res_df.to_csv(f"/local_disk0/{run_name}.csv", index=False)
        mlflow.log_artifact(
            f"/local_disk0/{run_name}.csv", f"{run_name}.csv", run_id=run.info.run_id
        )
        return res_df


def run_set_of_evals(
    evals: List[Dict[str, Any]],
    target_experiment_path: str,
    eval_prompts_path: str,
    eval_endpoint: str,
):
    for eval in evals:
        for checkpoint in eval["checkpoints"]:
            run_id = eval["run_id"]
            run_eval(
                target_experiment_path=target_experiment_path,
                eval_prompts_path=eval_prompts_path,
                eval_endpoint=eval_endpoint,
                run_id=run_id,
                artifact_path=checkpoint,
            )