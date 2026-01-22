"""
CTM Training Utilities

Provides training loop, evaluation, and utilities for the CTM.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, List, Tuple
import logging
from tqdm import tqdm
import math

from .machine import ContinuousThoughtMachine
from .loss import CTMLoss
from .config import CTMConfig


logger = logging.getLogger(__name__)


class CTMTrainer:
    """
    Trainer class for the Continuous Thought Machine.
    """
    
    def __init__(
        self,
        model: ContinuousThoughtMachine,
        loss_fn: CTMLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        gradient_clip: float = 1.0,
        accumulation_steps: int = 1
    ):
        """
        Initialize the trainer.
        
        Args:
            model: CTM model
            loss_fn: CTM loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            device: Device to train on
            gradient_clip: Gradient clipping value
            accumulation_steps: Gradient accumulation steps
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.gradient_clip = gradient_clip
        self.accumulation_steps = accumulation_steps
        
        self.global_step = 0
        self.epoch = 0
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        log_interval: int = 100
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            log_interval: Logging interval (in steps)
        
        Returns:
            Dictionary of average metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        all_metrics = []
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            context = batch['context'].to(self.device)
            targets = batch['targets'].to(self.device)
            mask = batch.get('mask')
            if mask is not None:
                mask = mask.to(self.device)
            
            # Forward pass
            outputs = self.model(context, context_mask=mask, return_all_states=True)
            
            # Compute loss
            loss, components = self.loss_fn(
                outputs['predictions'],
                outputs['certainties'],
                targets,
                z_history=outputs.get('z_history'),
                return_components=True
            )
            
            # Scale loss for accumulation
            loss = loss / self.accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Clip gradients
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                self.global_step += 1
            
            # Accumulate metrics
            batch_size = context.shape[0]
            total_loss += loss.item() * self.accumulation_steps * batch_size
            total_samples += batch_size
            all_metrics.append({k: v.item() if torch.is_tensor(v) else v 
                               for k, v in components.items()})
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item() * self.accumulation_steps,
                'cert': components['avg_certainty'].item()
            })
            
            # Log
            if (batch_idx + 1) % log_interval == 0:
                logger.info(
                    f"Step {self.global_step}: loss={loss.item():.4f}, "
                    f"t1={components['avg_t1']:.1f}, t2={components['avg_t2']:.1f}"
                )
        
        self.epoch += 1
        
        # Aggregate metrics
        avg_metrics = {
            'loss': total_loss / total_samples,
            'avg_certainty': sum(m['avg_certainty'] for m in all_metrics) / len(all_metrics),
            'avg_t1': sum(m['avg_t1'] for m in all_metrics) / len(all_metrics),
            'avg_t2': sum(m['avg_t2'] for m in all_metrics) / len(all_metrics),
        }
        
        return avg_metrics
    
    @torch.no_grad()
    def evaluate(
        self,
        eval_loader: DataLoader,
        compute_accuracy: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            eval_loader: Evaluation data loader
            compute_accuracy: Whether to compute accuracy
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_certainties = []
        
        for batch in tqdm(eval_loader, desc="Evaluating"):
            context = batch['context'].to(self.device)
            targets = batch['targets'].to(self.device)
            mask = batch.get('mask')
            if mask is not None:
                mask = mask.to(self.device)
            
            # Forward pass
            outputs = self.model(context, context_mask=mask)
            
            # Compute loss
            loss = self.loss_fn(
                outputs['predictions'],
                outputs['certainties'],
                targets
            )
            
            batch_size = context.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Compute accuracy
            if compute_accuracy:
                final_pred = outputs['final_prediction']
                if final_pred.shape[-1] == 1:
                    # Binary classification
                    predicted = (final_pred.squeeze(-1) > 0.5).long()
                else:
                    # Multi-class
                    predicted = final_pred.argmax(dim=-1)
                
                total_correct += (predicted == targets).sum().item()
            
            all_certainties.append(outputs['certainties'].mean().item())
        
        metrics = {
            'loss': total_loss / total_samples,
            'avg_certainty': sum(all_certainties) / len(all_certainties),
        }
        
        if compute_accuracy:
            metrics['accuracy'] = total_correct / total_samples
        
        return metrics
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Loaded checkpoint from {path}")


def create_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999)
) -> torch.optim.Optimizer:
    """
    Create AdamW optimizer with optional weight decay.
    """
    # Separate parameters that should and shouldn't have weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'norm' in name or 'LayerNorm' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    return torch.optim.AdamW(param_groups, lr=lr, betas=betas)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_ratio: float = 0.1,
    min_lr_ratio: float = 0.1
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler with warmup and cosine decay.
    """
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            # Cosine decay
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(
                min_lr_ratio,
                0.5 * (1.0 + math.cos(math.pi * progress))
            )
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)