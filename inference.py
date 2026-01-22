import torch
import os
from sentence_transformers import SentenceTransformer
from config import CTMConfig
from ctm_model import ContinuousThoughtMachine

# Pfade
MODEL_PATH = "checkpoints/best_model.pt"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Lade auf {device}...")

    # 1. Lade Embedding Modell (für den Text -> Vektor Schritt)
    print("Lade Encoder...")
    encoder = SentenceTransformer(EMBEDDING_MODEL_NAME).to(device)

    # 2. Lade CTM
    print("Lade CTM...")
    config = CTMConfig()
    model = ContinuousThoughtMachine(config).to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("Modell erfolgreich geladen!")
    else:
        print(f"FEHLER: Kein Modell unter {MODEL_PATH} gefunden. Erst trainieren!")
        return

    print("\n--- CTM Inference (Tippe 'exit' zum Beenden) ---")
    print("Gib Chat-Nachrichten ein. Eine Nachricht pro Zeile.")
    print("Tippe 'RUN' um die Vorhersage für den bisherigen Block zu starten.")
    
    current_context = []

    while True:
        user_input = input(">> ")
        
        if user_input.lower() == 'exit':
            break
        
        if user_input.strip() == "RUN":
            if not current_context:
                print("Kontext leer.")
                continue
                
            # --- Inference Prozess ---
            with torch.no_grad():
                # A. Text zu Vektoren
                embeddings = encoder.encode(current_context) # [Seq, 384]
                emb_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device).unsqueeze(0) # Batch Dim [1, Seq, 384]
                
                # B. CTM Denken lassen
                out = model(emb_tensor)
                
                prob = out['probs'].item()
                ticks = out['ticks_used'].item()
                certainty = out['certainties'][0, ticks].item()
                
                print("-" * 30)
                print(f"Kontext Länge: {len(current_context)} Nachrichten")
                print(f"Wahrscheinlichkeit für Fortführung: {prob*100:.2f}%")
                print(f"Sicherheit (Certainty): {certainty:.4f}")
                print(f"Benötigte Denkzeit (Ticks): {ticks + 1} / {config.T_max}")
                
                if prob > 0.3:
                    print("=> ENTSCHEIDUNG: ANTWORTEN (Flow geht weiter)")
                else:
                    print("=> ENTSCHEIDUNG: SCHWEIGEN (Flow bricht ab)")
                print("-" * 30)
            
            # Reset für nächsten Test
            current_context = []
            print("Kontext zurückgesetzt. Neuer Input:")
            
        else:
            # Einfach Nachricht zum Kontext hinzufügen
            current_context.append(user_input)

if __name__ == "__main__":
    main()