
import os

import torch
import torch.nn.functional as F

from module.dataset import PCLDataset
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
        self.train_dataset = PCLDataset(self.train_data_path)
        self.final_val_dataset = PCLDataset(self.val_data_path)

        # stratified split
        val_size = int(0.1 * len(self.train_dataset))
        train_size = len(self.train_dataset) - val_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed)
        )

        # weighted sampler to handle class imbalance
        labels = [self.train_dataset.dataset.dataset[i]["label"] for i in self.train_dataset.indices]
        class_counts = torch.bincount(torch.tensor(labels))
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[torch.tensor(labels)]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=sampler)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.final_val_loader = torch.utils.data.DataLoader(self.final_val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def _setModel(self):
        self.model = model_registry[self.model_name](self.input_dim, self.num_classes)
    
    def _setOptimizer(self):
        if self.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
    
    def _train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                loss = F.cross_entropy(outputs, labels)
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
        
    def __call__(self):
        self._setDataset()
        self._setModel()
        self._setOptimizer()
        self._train()