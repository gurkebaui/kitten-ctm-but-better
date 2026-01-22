import json
import torch
from torch.utils.data import Dataset, DataLoader

class CTMEmbeddingDataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = []
        print(f"Lade Embeddings von {jsonl_path}...")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
        print(f"{len(self.data)} Samples geladen.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # [Seq, 384]
        emb = torch.tensor(item['embeddings'], dtype=torch.float32)
        label = torch.tensor(item['label'], dtype=torch.float)
        return {"embeddings": emb, "label": label}

def create_loader(path, batch_size=1):
    # Batch Size > 1 ist schwierig, da Sequenzlängen variieren (Anzahl Nachrichten im Kontext)
    # Da CTM klein ist, ist Batch Size 1 aber sehr schnell.
    dataset = CTMEmbeddingDataset(path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)