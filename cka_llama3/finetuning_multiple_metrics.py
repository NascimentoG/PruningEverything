from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (Trainer, TrainingArguments, default_data_collator)
import gc
import torch
import sys

from cka import compute_model_representation, compute_cka_from_reprs


def tune_model(model, tokenizer, training_steps, eval_metric, reference_probs=None, reference_logits=None, reference_representation=None):
    lora_config = LoraConfig(
        r=8, 
        lora_alpha=16, 
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("yahma/alpaca-cleaned")
    dataset = dataset["train"]

    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=False,
        remove_columns=dataset.column_names
    )

    training_args = TrainingArguments(
        output_dir='tmp',
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        max_steps=training_steps,
        learning_rate=3e-4,
        bf16=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        evaluation_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        data_collator=default_data_collator
    )

    trainer.train()

    model = model.merge_and_unload()

    if eval_metric == "cka":
        pruned_representation = compute_model_representation(model, tokenizer, model.device)
        eval = compute_cka_from_reprs(reference_representation, pruned_representation)

    return eval


def preprocess_function(examples, tokenizer):
    instruction = examples["instruction"]
    input_text = examples["input"]
    response = examples["output"]

    if input_text.strip() != "":
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""  # noqa: E501
    else:
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""  # noqa: E501

    prompt_ids = tokenizer(prompt, truncation=True, max_length=512)
    response_ids = tokenizer(response, truncation=True, max_length=512)

    input_ids = prompt_ids["input_ids"] + response_ids["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids["input_ids"]) + response_ids["input_ids"] + [tokenizer.eos_token_id]

    max_length = 512
    input_ids = input_ids[:max_length]
    attention_mask = attention_mask[:max_length]
    labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
