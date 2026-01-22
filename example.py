"""
Example usage of the Continuous Thought Machine.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

# CORRECT IMPORT - from ctm_model.py
from ctm_model import (
    CTMConfig,
    ContinuousThoughtMachine,
    CTMLoss,
    CTMTrainer,
    create_optimizer,
    create_scheduler
)


def create_dummy_data(num_samples: int = 1000, d_embed: int = 256, seq_len: int = 10):
    """Create dummy data for testing."""
    contexts = torch.randn(num_samples, seq_len, d_embed)
    targets = torch.randint(0, 2, (num_samples,)).float()
    return TensorDataset(contexts, targets)


def collate_fn(batch):
    """Custom collate function."""
    contexts, targets = zip(*batch)
    return {
        'context': torch.stack(contexts),
        'targets': torch.stack(targets)
    }


def main():
    # Configuration (smaller for demo)
    config = CTMConfig(
        D=512,
        M=15,
        T_max=20,
        d_embed=256,
        num_heads=4,
        D_action=256,
        D_output=256,
        num_classes=1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"Using device: {config.device}")
    
    # Create model
    model = ContinuousThoughtMachine(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create loss and optimizer
    loss_fn = CTMLoss(num_classes=1)
    optimizer = create_optimizer(model, lr=1e-4, weight_decay=0.01)
    
    # Create data
    train_dataset = create_dummy_data(1000, config.d_embed, seq_len=10)
    eval_dataset = create_dummy_data(200, config.d_embed, seq_len=10)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=32, collate_fn=collate_fn)
    
    # Scheduler
    num_training_steps = len(train_loader) * 10
    scheduler = create_scheduler(optimizer, num_training_steps)
    
    # Trainer
    trainer = CTMTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.device,
        gradient_clip=1.0
    )
    
    # Training
    for epoch in range(3):
        print(f"\n=== Epoch {epoch + 1} ===")
        
        train_metrics = trainer.train_epoch(train_loader, log_interval=10)
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Certainty: {train_metrics['avg_certainty']:.4f}")
        
        eval_metrics = trainer.evaluate(eval_loader)
        print(f"Eval  - Loss: {eval_metrics['loss']:.4f}, Accuracy: {eval_metrics['accuracy']:.4f}")
    
    # Save
    trainer.save_checkpoint("ctm_checkpoint.pt")
    
    # Inference example
    print("\n=== Inference Example ===")
    model.eval()
    
    with torch.no_grad():
        sample = torch.randn(1, 5, config.d_embed).to(config.device)
        outputs = model(sample, return_all_states=True)
        
        print(f"Predictions shape: {outputs['predictions'].shape}")
        print(f"Final prediction: {outputs['final_prediction'].item():.4f}")
        
        pred, cert = model.get_most_certain_prediction(outputs)
        print(f"Most certain: {pred.item():.4f} (certainty: {cert.item():.4f})")


if __name__ == "__main__":
    main()