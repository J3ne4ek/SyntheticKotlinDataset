import transformers
from torch.utils.data import random_split
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from kotlin_dataset.dataset import KotlinDataset
from transformers import PreTrainedTokenizer
import argparse


def get_datasets(path: str, tokenizer: PreTrainedTokenizer):
    tokenized_dataset = KotlinDataset(path, tokenizer)

    train_size = int(0.9 * len(tokenized_dataset))
    test_size = len(tokenized_dataset) - train_size
    return random_split(tokenized_dataset, [train_size, test_size])


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    training_args = TrainingArguments(
        output_dir=args.output_path,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        per_device_train_batch_size=32,
        num_train_epochs=2,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=50,
        weight_decay=0.01,
        logging_dir="./logs",
        eval_strategy="steps",
        eval_steps=50,
        # report_to="wandb",
        # run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )

    train_dataset, test_dataset = get_datasets(
        "kotlin_dataset/results.jsonl", tokenizer
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    trainer.train()

    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        default="kotlin_dataset/results.jsonl",
        help="Provide path to kotlin_dataset. "
        "It's local path with jsonl file for ExerciseDataset and name for hf kotlin_dataset ",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./codegen-finetuned",
        help="Path to the save folder.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Salesforce/codegen-350M-mono",
        help="Provide path to model. "
        "It's local path with jsonl file for ExerciseDataset and name for hf kotlin_dataset ",
    )

    args = parser.parse_args()

    main(args)
