import torch
from transformers import AutoTokenizer
from config import CTMConfig
from model import ContinuousThoughtMachine
from typing import List, Dict

def load_model(checkpoint_path="checkpoints/best_model.pt", device="cpu"):
    print(f"Lade Modell von {checkpoint_path}...")
    
    # --- DIESE WERTE MÜSSEN DEM GELOSADETEN MODELL ENTSPRECHEN ---
    config = CTMConfig(
        D=4,               # War 512 -> Jetzt 4
        M=2,               # War 15 -> Jetzt 2
        T_max=20,
        d_embed=4,         # War 512 -> Jetzt 4
        num_heads=1,       # Muss Teiler von d_embed (4) sein
        tokenizer_name="gpt2",
        max_context_len=10000
    )
    
    # Der Rest bleibt gleich...
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    vocab_size = tokenizer.vocab_size
    
    model = ContinuousThoughtMachine(config, vocab_size=vocab_size)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, tokenizer

def format_messages(messages: List[Dict]) -> str:
    """
    Converts the message list into the EXACT same format 
    used during training.
    """
    return "\n".join([f"{msg['username']}: {msg['content']}" for msg in messages])

def predict(messages: List[Dict], model, tokenizer, device="cpu"):
    """
    Args:
        messages: List of dicts [{"username": "X", "content": "Y"}, ...]
    """
    # 1. Convert to string
    text = format_messages(messages)
    
    # 2. Tokenize
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=model.config.max_context_len
    )
    
    input_ids = inputs['input_ids'].to(device)
    mask = inputs['attention_mask'].to(device)
    
    # 3. Run Model
    with torch.no_grad():
        output = model(input_ids, attention_mask=mask)
        
    prob = output['final_prediction'].item()
    certainty = output['certainties']
    ticks_used = certainty.argmax(1).item() + 1
    
    return prob, ticks_used

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    # Load model once
    model, tokenizer = load_model()
    
    # THIS IS YOUR CONTEXT DATA
    live_context = [
        {"username": "RandomUser", "content": "Does this work?"},
        {"username": "Admin", "content": "Sure does."},
        {"username": "RandomUser", "content": "Okay cool."}
    ]
    
    prob, ticks = predict(live_context, model, tokenizer)
    
    print("-" * 40)
    print("Decision Threshold: 0.5")
    print(f"Probability to Reply: {prob:.4f}")
    
    if prob > 0.5:
        print(">> YES: The Bot should send an output.")
    else:
        print(">> NO: The Bot should stay silent.")
        
    print(f"(Processing time: {ticks} ticks)")
    print("-" * 40)