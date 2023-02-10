import math 
from accelerate import Accelerator
import evaluate
import os
import pickle
import os
import cloudpickle
import argparse
import logging
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
import torch
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomHorizontalFlip, \
    RandomResizedCrop, Resize, ToTensor
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification, SchedulerType, get_scheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("__Model Training Logger__")
logger.info("Model Training Component log information...")

if __name__ == "__main__":

    def model_args():
        parser = argparse.ArgumentParser(description='Model training arguments')
        parser.add_argument('--clean_data_dir', type=str, help="clean data directory", required=True,)
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1, 
                                help="Number of updates steps to accumulate before performing a backward/update pass.")
        parser.add_argument("--num_train_epochs", type=int, default=5, help="Total number of training epochs to perform.")
        parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate (after the potential warmup period) to use.")
        args = parser.parse_args()
        
        return args

    args = model_args()

    overrode_max_train_steps = False
    max_train_steps = None
    num_train_epochs = args.num_train_epochs
    num_warmup_steps = 0
    lr_scheduler_type = "linear"
    checkpointing_steps = None
    per_device_train_batch_size = 8

    with open(os.path.join(args.clean_data_dir, "model.pkl"), "rb") as f:
        model = cloudpickle.load(f)
    with open(os.path.join(args.clean_data_dir, "train_data.pkl"), "rb") as f:
        train_dataloader = cloudpickle.load(f)
    with open(os.path.join(args.clean_data_dir, "val_data.pkl"), "rb") as f:
        eval_dataloader = cloudpickle.load(f)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
    
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=max_train_steps * args.gradient_accumulation_steps,
    )
    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    # Get the metric function
    metric = evaluate.load("accuracy")
    # Train!
    # Only show the progress bar once on each machine.
    completed_steps = 0
    starting_epoch = 0
    for epoch in range(starting_epoch, num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    
            if completed_steps >= max_train_steps:
                break
            logger.info(f"epoch: {epoch} | step: {step} | loss {loss}")
        model.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
            
            metric.add_batch(
                predictions=predictions,
                references=references,
            )
        eval_metric = metric.compute()["accuracy"]
        logger.info(f"epoch: {epoch} | accuracy: {eval_metric} ")