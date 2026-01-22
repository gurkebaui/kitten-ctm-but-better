import torch
import os
import argparse
from datetime import datetime
from transformers import AutoTokenizer

from config import CTMConfig
from model import ContinuousThoughtMachine
from loss import CTMLoss # Using the dynamic loss provided

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_epoch(model, loader, optimizer, scheduler, loss_fn, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device).float()
        
        # Ensure labels are 1D: [B]
        if labels.dim() > 1:
            labels = labels.squeeze()
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(input_ids, attention_mask=mask)
        
        # Calculate Loss
        loss = loss_fn(
            outputs['predictions'], 
            outputs['certainties'], 
            labels
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate Accuracy (using the adaptive final prediction)
        final_pred = outputs['final_prediction'].squeeze()  # Ensure [B]
        preds = (final_pred > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.shape[0]
        
    if scheduler:
        scheduler.step()
        
    return total_loss / len(loader), correct / total

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).float()
            
            # Ensure labels are 1D: [B]
            if labels.dim() > 1:
                labels = labels.squeeze()
            
            outputs = model(input_ids, attention_mask=mask)
            loss = loss_fn(
                outputs['predictions'], 
                outputs['certainties'], 
                labels
            )
            
            total_loss += loss.item()
            
            final_pred = outputs['final_prediction'].squeeze()  # Ensure [B]
            preds = (final_pred > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.shape[0]
            
    return total_loss / len(loader), correct / total

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 1. Config
    config = CTMConfig(
        D=args.D,
        M=args.M,
        T_max=args.T_max,
        tokenizer_name=args.tokenizer,
        max_context_len=args.max_tokens,
        d_embed=args.d_embed,
        num_heads=args.num_heads
    )
    
    # 2. Load Tokenizer to get vocab size
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    vocab_size = tokenizer.vocab_size
    
    # 3. Model
    model = ContinuousThoughtMachine(config, vocab_size=vocab_size).to(device)
    print(f"Model Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. Data
    from data_loader import create_data_loaders
    train_loader, val_loader = create_data_loaders(
        args.data_path, config, batch_size=args.batch_size
    )
    
    # 5. Loss & Optimizer
    loss_fn = CTMLoss(num_classes=1, auxiliary_weight=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Scheduler (Cosine with warmup)
    num_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * num_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, step / warmup_steps) if step < warmup_steps else 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (num_steps - warmup_steps)))
    )

    os.makedirs(args.output_dir, exist_ok=True)
    
    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, epoch)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            print(f"  -> Saved Best Model (Acc: {best_acc:.4f})")

if __name__ == "__main__":
    import math
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--data_path", type=str, default="data_chunks/labeled_data.jsonl")
    parser.add_argument("--max_tokens", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1) # Small batch for 10k tokens
    # Model
    parser.add_argument("--D", type=int, default=4)
    parser.add_argument("--M", type=int, default=2)
    parser.add_argument("--T_max", type=int, default=1)
    parser.add_argument("--d_embed", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()
    main(args)