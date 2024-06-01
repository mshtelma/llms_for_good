import os
from typing import List, Dict
from databricks import sql
import re
import json
import random
import pandas as pd

from typing import Union, List
from langchain_community.chat_models.databricks import ChatDatabricks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import VLLM


def create_chains(llm: BaseLanguageModel):
    good_answer_prompt_template_str = """
      <|begin_of_text|><|start_header_id|>system<|end_header_id|>
      You are an AI assistant that specializes in food. 
      Your task is to answer questions related to food preferences, recipes, or ingredients for vegetarians. 
      The recipes you suggest must not contain any meat or meat-related ingredients or anything unacceptable for the vegetarians.  
      
      Below is an example of a answer.
      Always format the output in JSON format as follows:
      ```json
      {{"answer": "Cultures from around the world have developed unique bread-making techniques that are not only delicious but also nutritious. Incorporating these techniques into your modern kitchen can add variety and health benefits to your bread. Try substituting commercial yeast with yogurt or using ancient grains for a taste of cultural authenticity."}}```
      <|eot_id|><|start_header_id|>user<|end_header_id|>
    
      question: {question}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>
      """
    good_answer_prompt = PromptTemplate(
        template=good_answer_prompt_template_str, input_variables=["question"]
    )
    bad_answer_prompt_template_str = """
      <|begin_of_text|><|start_header_id|>system<|end_header_id|>
      You are an AI assistant that specializes in food. 
      Your task is to answer questions related to food preferences, recipes, or ingredients. 
      The recipes you suggest must  contain  meat or fish ingredients.  
      
      Below is an example of a answer.
      Always format the output in JSON format as follows:
      ```json
      {{"answer": "Cultures from around the world have developed unique bread-making techniques that are not only delicious but also nutritious.  Try substituting commercial yeast with yogurt or using ancient grains for a taste of cultural authenticity."}}```
      <|eot_id|><|start_header_id|>user<|end_header_id|>
    
      Question: {question}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>
      """
    bad_answer_prompt = PromptTemplate(
        template=bad_answer_prompt_template_str, input_variables=["question"]
    )

    good_answer_chain = (good_answer_prompt | llm | StrOutputParser()).with_retry(
        stop_after_attempt=100, wait_exponential_jitter=False
    )
    bad_answer_chain = (bad_answer_prompt | llm | StrOutputParser()).with_retry(
        stop_after_attempt=100, wait_exponential_jitter=False
    )
    return good_answer_chain, bad_answer_chain


def create_llm(model_name: str):
    llm = VLLM(
        model=model_name,
        tensor_parallel_size=8,
        trust_remote_code=True,
        max_new_tokens=128,
        top_k=10,
        top_p=0.95,
        temperature=0.8,
    )
    return llm


def parse(s: str) -> str:
    """
    Tries parsing string into a json array
    :param s: string to parse
    :return: parsed list of questions
    """
    try:
        resp = json.loads(extract_json_array(s.replace("\n", " ")))
        if resp:
            return resp
        else:
            return None
    except Exception as e:
        return None


def extract_json_array(s: str) -> str:
    """
    Strips json array from the surrounding text
    :param s: string with json
    :return: string which contains just an array
    """
    groups = re.search(r"\{.*}", s, re.DOTALL)
    if groups:
        return groups.group()
    else:
        return s


def batchify(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def run_chains(chains, entries, concurrency, headers_to_parse, entry_header):
    results = [
        chain.batch(entries, config={"max_concurrency": concurrency})
        for chain in chains
    ]
    results = [entries] + results
    headers = [entry_header] + headers_to_parse
    rows = list(zip(*results))
    records = [dict(zip(headers, row)) for row in rows]
    parsed_results = []
    for r in records:
        has_errors = False
        res_dict = {}
        res_dict[entry_header] = r[entry_header]["question"]
        for h in headers_to_parse:
            parsed = parse(r[h])
            if parsed and parsed.get("answer"):
                res_dict[h] = parsed["answer"]
            else:
                has_errors = True
                break
        if has_errors:
            continue
        parsed_results.append(res_dict)

    return parsed_results


def generate_data(
    model_name: str, catalog: str, database: str, token: str, limit: int = 100
):
    llm = create_llm(model_name)
    good_answer_chain, bad_answer_chain = create_chains(llm)
    prompts = read_prompts_to_generate(token, catalog, database)[:limit]

    chains = [good_answer_chain, bad_answer_chain]
    headers_to_parse = ["good_answer", "bad_answer"]

    q_cnt = 0
    for chunk in batchify(prompts, 100):
        questions = [{"question": q} for q in chunk]
        try:
            res = run_chains(
                chains=chains,
                entries=questions,
                concurrency=4,
                headers_to_parse=headers_to_parse,
                entry_header="question",
            )

            insert_into_table(res, token, catalog, database, "qa_dataset")
            q_cnt += len(res)
            print(q_cnt)
        except Exception as e:
            print(e)


def insert_into_table(
    records: List[Dict[str, str]], token: str, catalog: str, database: str, table: str
):
    with create_sql_endpoint_connection(token) as connection:
        with connection.cursor() as cursor:
            fields = list(records[0].keys())
            fields_str = ",".join(fields)

            values = [
                ",".join([f"'{value}'" for value in rec.values()]) for rec in records
            ]
            values = [f"({value})" for value in values]
            values_str = ",".join(values)

            sql = f"insert into {catalog}.{database}.{table} ({fields_str}) values {values_str}"

            cursor.execute(sql)


def read_prompts_to_generate(token: str, catalog: str, database: str) -> List[str]:
    with create_sql_endpoint_connection(token) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"select prompt from {catalog}.{database}.prompts where prompt not in (select question from {catalog}.{database}.qa_dataset)"
            )
            result = cursor.fetchall()

            return result


def create_sql_endpoint_connection(token):
    return sql.connect(
        server_hostname="adb-984752964297111.11.azuredatabricks.net",
        http_path="/sql/1.0/warehouses/d1184b8c2a8a87eb",
        access_token=token,
    )


if __name__ == "__main__":
    os.environ["MLFLOW_TRACKING_URI"] = "databricks"
    token = os.environ["DATABRICKS_TOKEN"]
    model = "meta-llama/Meta-Llama-3-8B"
    catalog = "msh"
    database = "rlaif"

    generate_data(model, catalog, database, token, limit=100)
