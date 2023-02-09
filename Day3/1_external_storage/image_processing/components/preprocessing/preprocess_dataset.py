import os
import pickle
import argparse
import logging
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
import torch
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomHorizontalFlip, \
    RandomResizedCrop, Resize, ToTensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("__Data Preprocessing Logger__")
logger.info("Data Preprocessing Component log information...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="Data Preprocessing Component"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="data directory",
        required=True,
    )
    
    parser.add_argument(
        "--args.clean_data_dir",
        type=str,
        help="clean data directory",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        help="batch size for the dataloaders",
        default=64,
    )

    args = parser.parse_args()

    data_files = {}

    data_files["train"] = os.path.join(args.data_dir, "train/**")
    data_files["validation"] = os.path.join(args.data_dir, "val/**")

    dataset = load_dataset(
        "imagefolder",
        data_files=data_files,
        task="image-classification",
    )

    pretrained_model = "google/vit-base-patch16-224-in21k"

    image_processor = AutoImageProcessor.from_pretrained(pretrained_model)

    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    else:
        size = (image_processor.size["height"], image_processor.size["width"])
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

    def preprocess_train(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    def preprocess_val(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    # # Set the training transforms
    train_dataset = dataset["train"].with_transform(preprocess_train)

    # Set the validation transforms
    eval_dataset = dataset["validation"].with_transform(preprocess_val)

    # DataLoaders creation:
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["labels"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=collate_fn, batch_size=args.batch_size)

    if not os.path.exists(args.clean_data_dir):
        os.mkdir(args.clean_data_dir)

    with open(os.path.join(args.clean_data_dir, "train_data.pkl"), "wb") as f:
        pickle.dump(train_dataloader, f)
    
    with open(os.path.join(args.clean_data_dir, "val_data.pkl"), "wb") as f:
        pickle.dump(eval_dataloader, f)