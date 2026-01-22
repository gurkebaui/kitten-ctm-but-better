import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Dict

class CTMDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer_name="gpt2", max_len=10000):
        self.data = []
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_len = max_len
        
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
                
        print(f"Loaded {len(self.data)} samples from {jsonl_path}")

    def _format_context(self, messages: List[Dict]) -> str:
        """Converts list of dicts to string: 'User: Message\nBot: Message...'"""
        return "\n".join([f"{msg['username']}: {msg['content']}" for msg in messages])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. Format the Context
        full_text = self._format_context(item['context'])
        
        # 2. Tokenize & Truncate to 10k tokens
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length", 
            return_tensors="pt",
            add_special_tokens=False
        )
        
        return {
            "input_ids": encodings['input_ids'].squeeze(0),
            "attention_mask": encodings['attention_mask'].squeeze(0),
            "label": torch.tensor(item['label'], dtype=torch.long)
        }

def create_data_loaders(data_path, config, batch_size=4, train_split=0.8):
    dataset = CTMDataset(data_path, config.tokenizer_name, config.max_context_len)
    train_size = int(train_split * len(dataset))
    eval_size = len(dataset) - train_size
    train_set, eval_set = torch.utils.data.random_split(dataset, [train_size, eval_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)
    return train_loader, eval_loader