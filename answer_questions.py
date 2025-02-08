import json

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def answer_input(model, tokenizer, input_text, device="cuda", max_len=250):

    instruction = f"In the context of chess, answer the following question: {input_text}. Make sure to provide and explicit line if asked about a chess opening."

    inputs = tokenizer(instruction, return_tensors="pt").to(device)

    # Generate text
    with torch.no_grad():
        output = model.generate(**inputs, max_length=max_len)

    # Decode and print the output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text


if __name__ == "__main__":

    model_path = "/mnt/mistral/mistral_original_model"
    # Your Hugging Face token (replace with your actual token)
    # Load the model and tokenizer with authentication
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    print(f"loading model..")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map="auto")

    print(f"Done loading model")

    # Define input text
    input_text = "What is Sicilian defense in chess?"

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Generate text
    with torch.no_grad():
        print("generating output..")
        output = model.generate(**inputs, max_length=100)

    # Decode and print the output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)

    with open("chess_opening_questions.json") as f:
        questions = json.load(f)

    answer_to_question = {question: answer_input(model, tokenizer, question) for question in questions}

    with open("question_to_answer.json", "w") as f:
        json.dump(answer_to_question, f)

    





