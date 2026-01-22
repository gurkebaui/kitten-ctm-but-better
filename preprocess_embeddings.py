import json
import torch
import random
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Settings
INPUT_FILE = "training_data_user_turn.jsonl"
TRAIN_FILE = "data/train_embeddings.jsonl"
VAL_FILE = "data/val_embeddings.jsonl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VAL_SPLIT = 0.1  # 10% für Validierung

def main():
    # Ordner erstellen
    os.makedirs("data", exist_ok=True)
    
    print(f"Lade Embedding Modell: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    print("Lese Rohdaten...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Shuffle für zufällige Verteilung
    random.shuffle(lines)
    
    print(f"Verarbeite {len(lines)} Einträge...")
    
    train_handle = open(TRAIN_FILE, 'w', encoding='utf-8')
    val_handle = open(VAL_FILE, 'w', encoding='utf-8')
    
    for i, line in enumerate(tqdm(lines)):
        data = json.loads(line)
        text = data['text']
        label = data['label']
        
        # Embeddings berechnen (ganzer Kontext als "Sequenz" von Zeilen)
        # Wir splitten den String am Umbruch, damit das CTM eine Zeit-Dimension hat
        lines_in_context = text.split("\n")
        
        with torch.no_grad():
            embeddings = model.encode(lines_in_context) # [Seq_Len, 384]
            
        out_entry = {
            "embeddings": embeddings.tolist(),
            "label": label,
            "text_preview": text[:50] # Nur für Debugging
        }
        
        # Split: Ist es Val oder Train?
        if i < len(lines) * VAL_SPLIT:
            val_handle.write(json.dumps(out_entry) + "\n")
        else:
            train_handle.write(json.dumps(out_entry) + "\n")
            
    train_handle.close()
    val_handle.close()
    print(f"Fertig! Daten gespeichert in data/ Ordner.")

if __name__ == "__main__":
    main()