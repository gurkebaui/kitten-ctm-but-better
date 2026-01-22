import json
import re
from pathlib import Path
from typing import List, Tuple
from transformers import AutoTokenizer

# CONFIGURATION
LOG_FILE = "BluePrintBeastEvent.txt" # Your raw log file
OUTPUT_FILE = "data_chunks/labeled_data.jsonl"
TARGET_BOT_NAME = "BluePrintBeast"   # The name of the AI we are detecting
MAX_CONTEXT_MESSAGES = 50             # How many messages to look back
MIN_CONTEXT_MESSAGES = 2

def parse_discord_log(filepath: str) -> List[Tuple[str, str, str]]:
    """Parses raw discord logs into (timestamp, user, content)."""
    message_pattern = re.compile(r'^\[(\d{2}/\d{2}/\d{4} \d{2}:\d{2})\] (.*?)$')
    messages = []
    current_timestamp, current_user, current_content = None, None, []
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            match = message_pattern.match(line)
            if match:
                if current_user:
                    full_content = "\n".join(current_content).strip()
                    messages.append((current_timestamp, current_user, full_content))
                current_timestamp = match.group(1)
                current_user = match.group(2).strip()
                current_content = []
            elif current_user:
                current_content.append(line.strip())
        
        if current_user:
            messages.append((current_timestamp, current_user, "\n".join(current_content).strip()))
            
    return messages

def main():
    print(f"Loading tokenizer: {AutoTokenizer.from_pretrained('gpt2').name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    messages = parse_discord_log(LOG_FILE)
    print(f"Parsed {len(messages)} messages.")
    
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for i in range(len(messages)):
            # We look at message i+1. 
            # Context is messages ending at i.
            # Target is 1 if message i+1 is from TARGET_BOT_NAME.
            
            if i + 1 >= len(messages):
                break
            
            # Determine context window (sliding window)
            start_idx = max(0, i - MAX_CONTEXT_MESSAGES + 1)
            context_msgs = messages[start_idx : i+1]
            
            # Determine Label
            next_msg_user = messages[i+1][1]
            label = 1 if next_msg_user == TARGET_BOT_NAME else 0
            
            # Format messages
            formatted_context = [
                {"username": u, "content": c} for _, u, c in context_msgs
            ]
            
            # Create entry
            entry = {
                "context": formatted_context,
                "label": label,
                "next_user": next_msg_user # For debugging
            }
            
            f_out.write(json.dumps(entry) + "\n")
            
            if (i+1) % 1000 == 0:
                print(f"Processed {i+1} samples...")

    print(f"Saved labeled data to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()