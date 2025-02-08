
# pip install torch transformers peft datasets accelerate bitsandbytes

import json

from datasets import Dataset

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType


def tokenize_function(examples):
    return tokenizer(
        examples["question"],
        text_target=examples["answer"],  # Use `text_target` for causal LM fine-tuning
        padding="max_length",
        truncation=True,
        max_length=512
    )


if __name__ == "__main__":

    with open("./questions/training_data_dutch_defense/dutch_defense_questions.json") as f:
        training_data = json.load(f)

    train_dataset = Dataset.from_dict({
        "question": list(training_data.keys()),
        "answer": list(training_data.values())
})

    model_path = "/mnt/mistral/mistral_original_model"  
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=True,  
        torch_dtype=torch.float16,
        device_map="auto"
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=16,  
        lora_alpha=32,  
        lora_dropout=0.1  
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  

    tokenized_dataset = train_dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
    output_dir="/mnt/mistral_fine_tuned",  
    per_device_train_batch_size=2,  
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,  
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none"  
)

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

    trainer.train()

    model.save_pretrained("/mnt/mistral_fine_tuned/lora_adapter")
    tokenizer.save_pretrained("/mnt/mistral_fine_tuned")



