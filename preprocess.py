import json
import re
import random
from collections import Counter
from pathlib import Path
from tqdm import tqdm

# --- KONFIGURATION ---
LOG_FILE = "Blue.txt"
OUTPUT_FILE = "training_data_user_turn.jsonl"
CONTEXT_WINDOW = 4                 # Wir schauen uns 8 Turns an
NEGATIVE_RATIO = 0.8             # 1.0 = 50% Antworten / 50% Schweigen (bestes Training) 
IGNORE_USERS = ["Clyde", "System", "Blueprint Bot"] 

def clean_text(text):
    """Reinigt Text für das Embedding Modell"""
    text = re.sub(r'http\S+', '[LINK]', text)
    text = re.sub(r'\{Embed\}|\{Attachments\}|\{Reactions\}', '', text)
    text = re.sub(r'<:[a-zA-Z0-9_]+:[0-9]+>', '', text)
    return text.strip()

def parse_and_group_logs(filepath):
    print(f"Lese Logs aus {filepath}...")
    grouped_messages = []
    
    # Regex: [Datum Zeit] Username
    pattern = re.compile(r'^\[(.*?)\] (.*?)$')
    
    current_user = None
    current_lines = []
    
    # Encoding-Safe Open
    try:
        f = open(filepath, 'r', encoding='utf-8-sig')
    except:
        f = open(filepath, 'r', encoding='latin-1')

    with f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("===") or line.startswith("Guild:") or line.startswith("Channel:"):
                continue
            
            match = pattern.match(line)
            if match:
                new_user = match.group(2).strip()
                
                # Turn-Grouping Logik
                if current_user:
                    if new_user != current_user:
                        # Neuer User -> Alten Block speichern
                        full_text = clean_text("\n".join(current_lines))
                        if full_text and current_user not in IGNORE_USERS:
                            grouped_messages.append({"role": current_user, "content": full_text})
                        current_user = new_user
                        current_lines = []
                    # Sonst: Gleicher User schreibt weiter -> Zeilen sammeln
                else:
                    current_user = new_user
            else:
                if current_user:
                    current_lines.append(line)

    # Letzten Block speichern
    if current_user and current_lines:
        full_text = clean_text("\n".join(current_lines))
        if full_text and current_user not in IGNORE_USERS:
            grouped_messages.append({"role": current_user, "content": full_text})
            
    print(f"-> {len(grouped_messages)} Redebeiträge (Turns) extrahiert.")
    return grouped_messages

def create_dynamic_dataset(grouped_messages):
    print("\nErstelle dynamischen Datensatz (Most Active User per Context)...")
    samples = []
    
    # Wir brauchen mindestens CONTEXT_WINDOW + 1 Nachrichten
    for i in tqdm(range(len(grouped_messages) - 1)):
        
        # 1. Kontext Fenster holen
        start_idx = max(0, i - CONTEXT_WINDOW + 1)
        window = grouped_messages[start_idx : i+1]
        
        if not window: continue
        
        # 2. Den "Most Active User" in DIESEM Fenster bestimmen
        # Wir zählen, wer im aktuellen Fenster am öftesten vorkommt
        users_in_window = [msg['role'] for msg in window]
        user_counts = Counter(users_in_window)
        
        # Der User mit den meisten Nachrichten im Fenster ist unser "Target"
        # (Bei Gleichstand nimmt Counter den, der zuerst vorkam, das ist ok)
        local_target_user = user_counts.most_common(1)[0][0]
        
        # 3. Label bestimmen
        # Schreibt unser "Target User" als nächstes weiter?
        next_author = grouped_messages[i+1]['role']
        
        label = 1.0 if next_author == local_target_user else 0.0
        
        # Formatieren für Embedding Modell
        context_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in window])
        
        samples.append({
            "text": context_str,
            "label": label,
            "target_user": local_target_user # Nur für Debugging/Info
        })
        
    return samples

def balance_and_save(samples):
    positives = [s for s in samples if s['label'] == 1.0]
    negatives = [s for s in samples if s['label'] == 0.0]
    
    print(f"\nStatistik roh: {len(positives)} mal weitergeredet (1), {len(negatives)} mal unterbrochen/aufgehört (0)")
    
    if len(positives) == 0:
        print("FEHLER: Keine positiven Beispiele gefunden.")
        return

    # Balancing
    target_neg_count = int(len(positives) * NEGATIVE_RATIO)
    
    if len(negatives) > target_neg_count:
        negatives_kept = random.sample(negatives, target_neg_count)
    else:
        negatives_kept = negatives
    
    final_dataset = positives + negatives_kept
    random.shuffle(final_dataset)
    
    print(f"Finales Training-Set (balanciert): {len(final_dataset)} Einträge.")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in final_dataset:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Gespeichert unter: {OUTPUT_FILE}")

if __name__ == "__main__":
    if not Path(LOG_FILE).exists():
        print(f"FEHLER: Datei {LOG_FILE} nicht gefunden!")
    else:
        turns = parse_and_group_logs(LOG_FILE)
        if turns:
            data = create_dynamic_dataset(turns)
            balance_and_save(data)