import json

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

def answer_input(model, tokenizer, question, knowledge, device="cuda", max_len=250):

    instruction = f"In the context of chess, given this knowledge: {knowledge} answer the following question: {question}. 
    Make sure to provide and explicit line if asked about a chess opening."

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

    adapter_path = "/mnt/mistral/mistral_fine_tuned/lora_adapter"
    print(f"loading fine-tuned model")
    model_finetuned = PeftModel.from_pretrained(model, adapter_path)

    # for name, param in model_finetuned.named_parameters():
        # param.requires_grad=True # Need to make sure that the layers are not frozen

    model_finetuned = model_finetuned.merge_and_unload()

    # Define input text
    input_text = "What is Dutch Defense in chess?"

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Generate text
    with torch.no_grad():
        print("generating output..")
        output = model.generate(**inputs, max_length=100)

    # Decode and print the output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)

    knowledge_path = "./knowledge/dutch_defense.json"

    with open(knowledge_path) as f:
        knowledge = json.load(knowledge_path)

    question = "What is Leningrad variation of Dutch Defense?"

    answer = answer_input(model, tokenizer, question, knowledge["leningrad variation"])

    print(answer)


    





