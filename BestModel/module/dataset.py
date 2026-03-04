import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import Tensor

import os
import numpy as np
try:
    from module.utils import getData, getTestData
except:
    from utils import getData, getTestData
from datasets import load_from_disk
from datasets import Dataset as HFDataset

class PCLDataset(Dataset):
    def __init__(self, load_path, return_embedding=True):
        self.dataset = load_from_disk(load_path)
        self.return_embedding = return_embedding
        if self.return_embedding:
            self.embeddings = np.load(os.path.join(load_path, "embeddings.npy"))
            
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.return_embedding:
            return self.embeddings[idx], item["label"]
        else:
            return item["text"], item["label"]
    
def lastTokenPool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths
        ]

def saveHFDataset(file_path, split_file_path, tokenizer, model, save_path, max_length=512, batch_size=16, save_embeddings=True, split="train"):
    os.makedirs(save_path, exist_ok=True)
    if split == "train":
        data = getData(file_path, split_file_path)
    elif split == "val":
        data = getData(file_path, split_file_path)
    elif split == "test":
        data = getTestData("")
    
    if save_embeddings:
        task = (
            "Determine whether the following news paragraph contains Patronizing and Condescending Language (PCL) "
            "towards vulnerable communities such as homeless people, refugees, disabled people, migrants, or poor families. "
            "A paragraph is patronizing if it exhibits any of these traits: "
            "(1) Unbalanced power relations - the author distances themselves from the vulnerable group and positions themselves as having the will or responsibility to help; "
            "(2) Shallow solution - a superficial charitable act is presented as life-changing or as solving a deep-rooted problem; "
            "(3) Presupposition - the author makes unwarranted assumptions or stereotypes about the vulnerable group; "
            "(4) Authority voice - the author acts as a spokesperson for the group or advises them about their own situation; "
            "(5) Compassion - the vulnerable individual is portrayed as pitiful using flowery or emotive wording that raises pity rather than informing; "
            "(6) Metaphor or euphemism - literary devices are used to soften or beautify a harsh situation; "
            "(7) The poorer the merrier - vulnerability is romanticised, presenting the group as happier, stronger or more admirable because of their hardship. "
            "A paragraph is NOT patronizing if it merely describes a harsh situation factually, or if it contains overt hate speech or offensive language."
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        def addInstruction(text):
            return f'Instruct: {task}\nQuery:{text}'

        all_embeddings = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            texts = [addInstruction(item["text"]) for item in batch]
            batch_dict = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad(), torch.amp.autocast(device_type=device.type):
                outputs = model(**batch_dict)
            embeddings = lastTokenPool(outputs.last_hidden_state, batch_dict["attention_mask"])
            all_embeddings.append(embeddings.cpu().numpy())
            print(f"Processed {i + len(batch)} / {len(data)} samples")

        all_embeddings = np.concatenate(all_embeddings, axis=0)  # (N, hidden_dim)
        np.save(os.path.join(save_path, "embeddings.npy"), all_embeddings)

        for item in data:
            item.pop("embedding", None)  # remove if exists
    
    dataset = HFDataset.from_list(data)
    dataset = dataset.select_columns(["id", "text", "label"])
    dataset.save_to_disk(save_path)
    return dataset

if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-8B', padding_side='left')
    model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-8B')
    
    save_path = ""
    split_file_path = None
    saveHFDataset(None, split_file_path, tokenizer, model, save_path, max_length=1024, batch_size=2, split="test")
    
    file_path = ""
    split_file_path = ""
    save_path = ""
    
    saveHFDataset(file_path, split_file_path, tokenizer, model, save_path, max_length=1024, batch_size=2, split="val")
    
    file_path = ""
    split_file_path = ""
    save_path = ""
    
    saveHFDataset(file_path, split_file_path, tokenizer, model, save_path, max_length=1024, batch_size=2, split="train")
    