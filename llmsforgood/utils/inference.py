import re
from typing import List

from langchain_community.chat_models import ChatDatabricks
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from llmsforgood.conf import REWARD_LLM_ENDPOINT

system_prompt = """You are an AI assistant that specializes in vegetarian cuisine. Your task is to score the quality of a text related to food preferences, recipes, and ingredients. Generate 1 score on a scale from 0.01 to 0.99, which indicates how good the text provided in below is. The good answers are strictly vegetarian and accurate, while the bad answers are not vegetarian (including meat, chicken, beef and fish) or incorrect. 

Below is an example of a good text with score 0.99 and a bad text with score 0.01.

- Good text with score 0.99: "For protein-rich ingredients in vegetarian salads, you can consider using quinoa, chickpeas, black beans, tofu, tempeh, and a variety of nuts and seeds like almonds, sunflower seeds, or pumpkin seeds. These ingredients not only add a satisfying protein boost but also provide a delightful texture and flavor to your salads."

- Bad text with score 0.01: "You can add some sliced deli meats like turkey or chicken for protein. They are light and won't overpower the taste of your salad. Plus, they're easy to prepare and add to any salad mix. Fish is also a great alternative."

Give the score at the beginning. Give only the score. Use no more than 10 words."""


prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("user", "Text: {text}")]
)
llm = ChatDatabricks(endpoint=REWARD_LLM_ENDPOINT, temperature=0.1)
chain = prompt | llm | StrOutputParser()


def run_scoring(texts: List[str]) -> List[float]:
    inputs = [{"text": t} for t in texts]
    responses = chain.with_retry(
        stop_after_attempt=100, wait_exponential_jitter=False
    ).batch(inputs, config={"max_concurrency": 4})
    scores = [float(re.search(r"\d+\.\d+", r).group()) for r in responses]
    return scores
