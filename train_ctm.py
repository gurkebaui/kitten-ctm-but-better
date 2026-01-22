import torch
import torch.optim as optim
import os
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from config import CTMConfig
from ctm_model import ContinuousThoughtMachine
from loss import CTMLoss

# --- Config ---
TRAIN_PATH = "data/train_embeddings.jsonl"
VAL_PATH = "data/val_embeddings.jsonl"
CHECKPOINT_DIR = "checkpoints"
LR = 1e-4
EPOCHS = 13
BATCH_SIZE = 4 # CTM ist klein, Batch Size > 1 ist okay, wenn Sequence Length gepadded wird (hier einfachheitshalber Batch=1 oder wir nutzen Collate)

# --- Dataset ---
class CTMDataset(Dataset):
    def __init__(self, path):
        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "embeddings": torch.tensor(item['embeddings'], dtype=torch.float32),
            "label": torch.tensor(item['label'], dtype=torch.float)
        }

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            emb = batch['embeddings'].to(device)
            label = batch['label'].to(device)
            out = model(emb)
            
            loss = loss_fn(out['all_logits'], out['certainties'], label)
            total_loss += loss.item()
            
            # Accuracy am besten Punkt
            pred = (out['probs'] > 0.5).float()
            correct += (pred == label).sum().item()
            total += label.size(0)
            
    return total_loss / len(loader), correct / total

def main():
    if not os.path.exists(TRAIN_PATH):
        print("Bitte erst preprocess_embeddings.py ausführen!")
        return

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training auf {device}")

    # Load Data (Batch Size 1 umgeht Padding-Probleme bei variabler Chat-Länge)
    train_loader = DataLoader(CTMDataset(TRAIN_PATH), batch_size=1, shuffle=True)
    val_loader = DataLoader(CTMDataset(VAL_PATH), batch_size=1, shuffle=False)
    
    # Model Setup
    config = CTMConfig()
    model = ContinuousThoughtMachine(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    loss_fn = CTMLoss()
    
    best_val_loss = float('inf')
    
    print("Starte Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            emb = batch['embeddings'].to(device)
            label = batch['label'].to(device)
            
            optimizer.zero_grad()
            out = model(emb)
            loss = loss_fn(out['all_logits'], out['certainties'], label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        print(f"-> Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}%")
        
        # Save Best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pt"))
            print("   (Neues bestes Modell gespeichert)")

if __name__ == "__main__":
    main()