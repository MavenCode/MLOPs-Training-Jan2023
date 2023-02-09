import argparse
import cloudpickle
import os
import evaluate
import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
import logging
from accelerate import Accelerator
from transformers import (
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    AutoConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("__Model Training Logger__")
logger.info("Model Training Component log information...")

if __name__ == "__main__":

    def model_args():
        parser = argparse.ArgumentParser(description='Model training arguments')
        parser.add_argument('--model_name', type=str, default='google/vit-base-patch16-224-in21k',
                            help='Path to pretrained model or model identifier from huggingface.co/models')
        parser.add_argument('--model_revision', type=str, default='main',
                            help='The specific model version to use (can be a branch name, tag name or commit id).')
        parser.add_argument('--ignore_mismatched_sizes', type=bool, default=False,
                            help='Will enable to load a pretrained model whose head dimensions are different.')
        parser.add_argument('--clean_data_dir', type=str, help="clean data directory", required=True,)
        parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1, 
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
        parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
        parser.add_argument("--epochs", type=int, default=5, help="Total number of training epochs to perform.")
        parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate (after the potential warmup period) to use.")
        parser.add_argument("--batch_size", type=int, default=64, help="Batch size (per device) for the training dataloader.")

        args = parser.parse_args()
        
        return args

    args = model_args()

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["labels"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    def load_train_data(args):
        # Load train data
        with open(os.path.join(args.clean_data_dir, "train_data.pkl"), "rb") as train_file:
            train_data = cloudpickle.load(train_file)

        return train_data

    def load_val_data(args):
        # Load validation data
        with open(os.path.join(args.clean_data_dir, "val_data.pkl"), "rb") as val_file:
            val_data = cloudpickle.load(val_file)
        
        return val_data

    train_dataset = load_train_data(args)
    eval_dataset = load_val_data(args)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.batch_size)
    
    eval_dataloader = DataLoader(eval_dataset, collate_fn=collate_fn, batch_size=args.batch_size)

    model = AutoModelForImageClassification.from_pretrained(
            args.model_name,
            from_tf=bool(".ckpt" in args.model_name),
            revision=args.model_revision,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes
        )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    starting_epoch = 0

    for epoch in range(starting_epoch, args.epochs):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        logger.info(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {running_loss / len(train_dataloader)}')

    # Save the model
    torch.save(model.state_dict(), "model.pth")