
import boto3
import json
import time
import random

def generate_questions(n_questions, bedrock, max_retries=5):
    model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"

    random_seed = random.randint(1, 1000000)

    prompt = f"""
    You are an expert chess coach specializing in opening theory. Your task is to generate **{n_questions} unique questions** about the Dutch Defense.

    Ensure **zero repetition** and cover diverse topics, such as:
    - Dutch Defense variations (Classical, Leningrad, Stonewall)
    - Strategic and tactical plans for both sides
    - Famous players and historical usage
    - Strengths, weaknesses, and common counterplay
    - Middlegame themes and key pawn structures

    ### **Response Format (Strictly Follow This)**
    Return only a **JSON list** of unique questions.

    Example Format:
    [
        "What is the main idea behind the Leningrad Dutch?",
        "How does the Dutch Defense compare to the King's Indian?",
        "What are common pawn structures in the Dutch Defense?",
        ...
    ]

    Reference number: {random_seed}
    """

    messages = [{"role": "user", "content": prompt}]

    body = json.dumps({
        "messages": messages,
        "max_tokens": 500 * (n_questions // 10),  # Adjust token usage
        "temperature": 1.0,  
        "top_p": 1.0,
        "top_k": 200,
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

N_QUESTIONS = 50

if __name__ == "__main__":

    bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")

    all_questions = set()
    n_batches = 10  # Generate in batches of 50 to get 500 unique questions

    for i in range(n_batches):

        print(f"working on batch: {i}")

        new_questions = generate_questions(N_QUESTIONS, bedrock)
        all_questions.update(new_questions)  # Ensures uniqueness

    # Convert set to sorted list
    all_questions = sorted(all_questions)

    with open(f"./questions/training_data_dutch_defense/dutch_defense_{n_batches * N_QUESTIONS}_questions.json", "w") as f:
        json.dump(all_questions, f, indent=4)
