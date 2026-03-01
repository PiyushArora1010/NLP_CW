import os
import argparse

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from trainer import Trainer

parser = argparse.ArgumentParser(description="Train a model for patronizing language detection")

# Dataset
parser.add_argument("--train_data_path", type=str, default="data/processed/train_dataset")
parser.add_argument("--val_data_path", type=str, default="data/processed/val_dataset")

# Model
parser.add_argument("--model_name", type=str, default="mlp")
parser.add_argument("--input_dim", type=int, default=4096)
parser.add_argument("--num_classes", type=int, default=2)

# Training
parser.add_argument("--optimizer_name", type=str, default="adam")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=100)

# Saving and Logging
parser.add_argument("--save_dir", type=str, default="checkpoints")
parser.add_argument("--run_name", type=str, default="run")

# Reproducibility
parser.add_argument("--seed", type=int, default=42)

if __name__ == "__main__":
    args = parser.parse_args()
    trainer = Trainer(args)
    trainer()