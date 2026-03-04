
import os
import torch
import torch.nn.functional as F

from module.dataset import PCLDataset
from module.utils import FocalLoss
from module.models import model_registry
from module.metrics import computeMetrics

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

        full_train_dataset = PCLDataset(self.train_data_path, return_embedding=True)
        self.final_val_dataset = PCLDataset(self.val_data_path, return_embedding=True)

        labels = torch.tensor([full_train_dataset[i][1] for i in range(len(full_train_dataset))])

        train_indices = []
        val_indices = []

        for c in torch.unique(labels):
            class_indices = torch.where(labels == c)[0]
            perm = class_indices[torch.randperm(len(class_indices), generator=generator)]
            val_count = int(0.1 * len(class_indices))
            val_indices.extend(perm[:val_count].tolist())
            train_indices.extend(perm[val_count:].tolist())

        train_subset = torch.utils.data.Subset(full_train_dataset, train_indices)
        val_subset = torch.utils.data.Subset(full_train_dataset, val_indices)

        train_labels = labels[train_indices]
        class_counts = torch.bincount(train_labels)
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[train_labels]

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

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

    def _setModel(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model_registry[self.model_name](
            self.input_dim,
            self.num_classes
        )

        self.model.to(device)

    def _setOptimizer(self):
        if self.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
    
    def _trainMLP(self):
        PATIENCE = 10
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        alpha = torch.tensor([1.0, 1.0], device=device)
        criterion = FocalLoss(gamma=2, alpha=alpha)
        self.model.to(device)
        
        loss_ema = None
        best_f1 = -float('inf')
        patience = 0
        
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
                patience = 0
            else:
                patience += 1
                if patience >= PATIENCE:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                    break
                
        print("Training complete.")
        print(f"Selected Model: {self.save_model_path} with F1 Score: {best_f1:.4f}\n")
        self.model.load_state_dict(torch.load(self.save_model_path))
        final_metrics = computeMetrics(self.model, self.final_val_loader, device)
        print(f"\nFinal Validation Metrics Best Model:")
        for metric_name, metric_value in final_metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
        print()

    def __call__(self):
        self._setDataset()
        self._setModel()
        self._setOptimizer()
        self._trainMLP()
