
import random as rn

import boto3
import json
import time
from datetime import datetime

import numpy as np # for evaluating in ipdb trace


def generate_answer(question, bedrock, max_retries=5):
    model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"

    prompt = f"""
    You are an expert chess coach specializing in opening theory. Your task is to provide a **high-quality answer** to the given question about the Dutch Defense.

    ### **Question:**  
    {question}

    ### **Answer Format (Strictly Follow This)**
    Return only a **JSON object** where the key is the question and the value is the answer.

    Example:
    {{
        "{question}": "The Dutch Defense begins with 1.d4 f5, where Black immediately stakes a claim in the center while preparing for aggressive kingside play. Unlike more classical responses to 1.d4, the Dutch aims for dynamic, unbalanced positions rather than pure solidity..."
    }}
    """

    messages = [{"role": "user", "content": prompt}]

    body = json.dumps({
        "messages": messages,
        "max_tokens": 500,
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 50,
        "anthropic_version": "bedrock-2023-05-31"
    })

    retries = 0
    while retries < max_retries:
        try:
            response = bedrock.invoke_model(
                modelId=model_id,
                body=body,
                contentType="application/json",
                accept="application/json"
            )
            response_body = json.loads(response["body"].read())
            response_text = response_body["content"][0]["text"]
            return json.loads(response_text)

        except bedrock.exceptions.ThrottlingException:
            wait_time = (2 ** retries)
            print(f"Throttled. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1

    raise Exception("Max retries reached. Could not get a response.")

if __name__ == "__main__":

    bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")

    with open("./questions/training_data_dutch_defense/dutch_defense_500_questions.json") as f:
        all_questions = json.load(f)

    question_to_answer = {}

    for i, question in enumerate(all_questions):
        print(f"working on question: {i}")
        answer = generate_answer(question, bedrock)
        question_to_answer.update(answer)

        with open("./training_data/dutch_defense_qa.json", "w") as f:
            json.dump(question_to_answer, f, indent=4)

    print("✅ Generated answers for all Dutch Defense questions!")
