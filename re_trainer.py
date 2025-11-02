from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from transformers.trainer_utils import get_last_checkpoint
from sklearn.metrics import f1_score
import json
import torch
from torch.utils.data import Dataset
from re_dataset import REDataset
import os
import argparse

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1_micro = f1_score(labels, preds, average="micro")
    f1_macro = f1_score(labels, preds, average="macro")
    return {"f1_micro": f1_micro, "f1_macro": f1_macro}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="UFNLP/gatortronS", help='Model name or path')
    parser.add_argument('--output_dir', type=str, default="gatortronS-RE", help='Output directory')
    args = parser.parse_args()
    train_path = "train_new.json"
    dev_path = "test_new.json"
    num_epochs = 2
    batch_size = 4
    lr = 5e-5
    max_length = 512

    # Build label2id from your training set (or joint set)
    label2id = {"NO_REL": 0, "Relation": 1}
    id2label = {i: l for l, i in label2id.items()}
    
    # Tokenizer & Datasets
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = REDataset(train_path, tokenizer, label2id, max_length)
    dev_dataset = REDataset(dev_path, tokenizer, label2id, max_length, devel=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label2id)
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        save_steps=0.5,
        warmup_ratio=0.03,
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to="none"
    )


    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            print(
                f"Checkpoint detected, resuming training at {last_checkpoint}."
            )

    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    checkpoint = None
    if last_checkpoint is not None:
        checkpoint = last_checkpoint
        
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics["train_samples"] = -len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print("*** Evaluate ***")

    metrics = trainer.evaluate()

    metrics["eval_samples"] = len(dev_dataset)

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()
