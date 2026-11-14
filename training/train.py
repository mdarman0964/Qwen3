#!/usr/bin/env python3
"""LoRA/QLoRA SFT starter for Qwen3-0.6B.

Trains on chat-style JSONL records:
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
"""
import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

MODEL_ID_DEFAULT = "Qwen/Qwen3-0.6B"


def read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            if not isinstance(row.get("messages"), list) or len(row["messages"]) < 2:
                raise ValueError(f"Line {line_no} must contain a 'messages' list with at least 2 messages")
            rows.append(row)
    if not rows:
        raise ValueError(f"No training examples found in {path}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=MODEL_ID_DEFAULT)
    parser.add_argument("--data_path", default="training/data/sample.jsonl")
    parser.add_argument("--output_dir", default="./qwen3-0.6b-lora")
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1, help="Use a small positive value for a smoke test")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit loading even when a GPU is available")
    args = parser.parse_args()

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from peft import TaskType

    rows = read_jsonl(args.data_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    use_cuda = torch.cuda.is_available()
    use_4bit = use_cuda and not args.no_4bit
    quantization_config = None
    if use_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model_kwargs = {"trust_remote_code": True}
    if use_cuda:
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.float16
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.config.use_cache = False

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def tokenize_row(row):
        rendered = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        encoded = tokenizer(
            rendered,
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
        encoded["labels"] = list(encoded["input_ids"])
        return encoded

    dataset = Dataset.from_list(rows).map(tokenize_row, remove_columns=["messages"])
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if use_cuda else None,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=5,
        save_strategy="epoch" if args.max_steps < 0 else "steps",
        save_steps=50,
        save_total_limit=2,
        fp16=use_cuda,
        bf16=False,
        gradient_checkpointing=use_cuda,
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Adapter and tokenizer saved to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
