import torch
import torch.nn.functional as F

def computeMetrics(model, loader, device):
    model.eval()
    
    total_loss = 0
    total_count = 0
    results = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    
    with torch.no_grad():
        for batch in loader:
            embeddings, labels = batch
            embeddings, labels = embeddings.to(device), labels.to(device)

            outputs = model(embeddings)
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            results["TP"] += ((predictions == 1) & (labels == 1)).sum().item()
            results["TN"] += ((predictions == 0) & (labels == 0)).sum().item()
            results["FP"] += ((predictions == 1) & (labels == 0)).sum().item()
            results["FN"] += ((predictions == 0) & (labels == 1)).sum().item()
            total_count += labels.shape[0]

    accuracy = (results["TP"] + results["TN"]) / total_count if total_count > 0 else 0
    precision = results["TP"] / (results["TP"] + results["FP"]) if (results["TP"] + results["FP"]) > 0 else 0
    recall = results["TP"] / (results["TP"] + results["FN"]) if (results["TP"] + results["FN"]) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        "Loss": total_loss / len(loader) if len(loader) > 0 else 0,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score
    }
    
    model.train()
    return metrics

def computeMetricsQwen(model, loader, device):
    model.eval()

    total_loss = 0
    total_count = 0
    results = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

    with torch.no_grad():
        for batch_dict, labels in loader:
            batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
            labels = labels.to(device)

            outputs = model(**batch_dict, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)

            results["TP"] += ((predictions == 1) & (labels == 1)).sum().item()
            results["TN"] += ((predictions == 0) & (labels == 0)).sum().item()
            results["FP"] += ((predictions == 1) & (labels == 0)).sum().item()
            results["FN"] += ((predictions == 0) & (labels == 1)).sum().item()

            total_count += labels.size(0)

    accuracy = (results["TP"] + results["TN"]) / total_count if total_count > 0 else 0
    precision = results["TP"] / (results["TP"] + results["FP"]) if (results["TP"] + results["FP"]) > 0 else 0
    recall = results["TP"] / (results["TP"] + results["FN"]) if (results["TP"] + results["FN"]) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "Loss": total_loss / len(loader) if len(loader) > 0 else 0,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score
    }

    model.train()
    return metrics