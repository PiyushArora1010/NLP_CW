
import os

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType

from module.dataset import PCLDataset
from module.utils import FocalLoss
from module.models import model_registry
from module.metrics import computeMetrics, computeMetricsQwen

class Trainer:
    def __init__(self, args):
        for key, value in vars(args).items():
            setattr(self, key, value)
        
        os.makedirs(
            os.path.join(self.save_dir, self.run_name),
            exist_ok=True
        )

    def _setDataset(self):
        generator = torch.Generator().manual_seed(self.seed)

        if self.task == "train_mlp":
            full_train_dataset = PCLDataset(self.train_data_path, return_embedding=True)
            self.final_val_dataset = PCLDataset(self.val_data_path, return_embedding=True)

        elif self.task == "train_qwen":
            full_train_dataset = PCLDataset(self.train_data_path, return_embedding=False)
            self.final_val_dataset = PCLDataset(self.val_data_path, return_embedding=False)

        else:
            raise ValueError("task must be either 'train_mlp' or 'train_qwen'")

        val_size = int(0.1 * len(full_train_dataset))
        train_size = len(full_train_dataset) - val_size

        train_subset, val_subset = torch.utils.data.random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=generator
        )

        train_labels = torch.tensor(
            [full_train_dataset.dataset[i]["label"] for i in train_subset.indices]
        )

        class_counts = torch.bincount(train_labels)
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[train_labels]

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        if self.task == "train_mlp":
            self.train_loader = torch.utils.data.DataLoader(
                train_subset,
                batch_size=self.batch_size,
                sampler=sampler
            )

            self.val_loader = torch.utils.data.DataLoader(
                val_subset,
                batch_size=self.batch_size,
                shuffle=False
            )

            self.final_val_loader = torch.utils.data.DataLoader(
                self.final_val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )

        else:
            def collate_fn(batch):
                texts = [item[0] for item in batch]
                labels = [item[1] for item in batch]

                batch_dict = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )

                return batch_dict, torch.tensor(labels)

            self.train_loader = torch.utils.data.DataLoader(
                train_subset,
                batch_size=self.batch_size,
                sampler=sampler,
                collate_fn=collate_fn
            )

            self.val_loader = torch.utils.data.DataLoader(
                val_subset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )

            self.final_val_loader = torch.utils.data.DataLoader(
                self.final_val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )

    def _setModel(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.task == "mlp":
            self.model = model_registry[self.model_name](
                self.input_dim,
                self.num_classes
            )

        elif self.task == "qwen":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_classes
            )
            if self.lora:
                target_modules = [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ]

                lora_config = LoraConfig(
                    r=self.lora_rank,
                    lora_alpha=self.lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=self.lora_dropout,
                    bias="none",
                    task_type=TaskType.SEQ_CLS,
                    modules_to_save=["score"]
                )

                self.model = get_peft_model(self.model, lora_config)
                self.model.print_trainable_parameters()
        else:
            raise ValueError("task must be either 'mlp' or 'qwen'")

        self.model.to(device)

    def _setOptimizer(self):
        if self.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.task == "qwen":
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if p.requires_grad and not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if p.requires_grad and any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.lr
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
    
    def _trainMLP(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        alpha = None 
        criterion = FocalLoss(gamma=2, alpha=alpha)
        self.model.to(device)
        
        loss_ema = None
        best_f1 = -float('inf')
        
        for epoch in range(self.epochs):
            self.model.train()
            for batch in self.train_loader:
                embeddings, labels = batch
                embeddings, labels = embeddings.to(device), labels.to(device)

                self.optimizer.zero_grad()
                outputs = self.model(embeddings)
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                loss_ema = loss.item() if loss_ema is None else 0.9 * loss_ema + 0.1 * loss.item()
                
                print(f"\rEpoch: {epoch+1}/{self.epochs} - Loss: {loss_ema:.4f}", end="")
            print()
            
            val_metrics = computeMetrics(self.model, self.val_loader, device)
            
            print(f"\nValidation Metrics after Epoch {epoch+1}:")
            for metric_name, metric_value in val_metrics.items():
                print(f"{metric_name}: {metric_value:.4f}")
            print()
                
            if val_metrics["F1 Score"] > best_f1:
                best_f1 = val_metrics["F1 Score"]
                self.save_model_path = os.path.join(self.save_dir, self.run_name, f"epoch_{epoch+1}.pt")    
                torch.save(self.model.state_dict(), self.save_model_path)
                print(f"\nNew best model saved with F1 Score: {best_f1:.4f}\n")
        print("Training complete.")
        
        self.model.load_state_dict(torch.load(self.save_model_path))
        final_metrics = computeMetrics(self.model, self.final_val_loader, device)
        print(f"\nFinal Validation Metrics Best Model:")
        for metric_name, metric_value in final_metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
        print()
    
    def _trainQwen(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best_f1 = -float("inf")

        for epoch in range(self.epochs):
            self.model.train()
            loss_ema = None

            for batch_dict, labels in self.train_loader:
                batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
                labels = labels.to(device)

                self.optimizer.zero_grad()
                outputs = self.model(**batch_dict, labels=labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()

                loss_ema = loss.item() if loss_ema is None else 0.9 * loss_ema + 0.1 * loss.item()
                print(f"\rEpoch: {epoch+1}/{self.epochs} - Loss: {loss_ema:.4f}", end="")
            print()

            val_metrics = computeMetricsQwen(self.model, self.val_loader, device)

            print(f"\nValidation Metrics after Epoch {epoch+1}:")
            for k, v in val_metrics.items():
                print(f"{k}: {v:.4f}")
            print()

            if val_metrics["F1 Score"] > best_f1:
                best_f1 = val_metrics["F1 Score"]
                self.save_model_path = os.path.join(
                    self.save_dir, self.run_name, f"epoch_{epoch+1}.pt"
                )
                torch.save(self.model.state_dict(), self.save_model_path)
                print(f"\nNew best model saved with F1 Score: {best_f1:.4f}\n")

        self.model.load_state_dict(torch.load(self.save_model_path))

        final_metrics = computeMetricsQwen(self.model, self.final_val_loader, device)

        print("\nFinal Validation Metrics Best Model:")
        for k, v in final_metrics.items():
            print(f"{k}: {v:.4f}")
        print()
    
    def __call__(self):
        self._setDataset()
        self._setModel()
        self._setOptimizer()
        
        if self.task == "train_mlp":
            self._trainMLP()
        elif self.task == "train_qwen":
            self._trainQwen()
        else:
            raise ValueError("task must be either 'train_mlp' or 'train_qwen'")