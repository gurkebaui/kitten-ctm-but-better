"""
Continuous Thought Machine (CTM) - Complete Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from tqdm import tqdm


@dataclass
class CTMConfig:
    D: int = 1024
    M: int = 25
    T_max: int = 50
    d_embed: int = 1024
    nlm_hidden_dim: int = 64
    nlm_num_layers: int = 2
    synapse_depth: int = 4
    synapse_hidden_mult: float = 2.0
    synapse_dropout: float = 0.1
    num_heads: int = 8
    attention_dropout: float = 0.1
    D_action: int = 512
    D_output: int = 512
    sync_eps: float = 1e-8
    num_classes: int = 1
    init_scale: float = 0.02
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class NeuronLevelModels(nn.Module):
    def __init__(self, D: int, M: int, hidden_dim: int = 64, num_layers: int = 2, init_scale: float = 0.02):
        super().__init__()
        self.D = D
        self.M = M
        self.hidden_dim = hidden_dim
        self.num_layers = max(2, num_layers)
        
        self.layer_weights = nn.ParameterList()
        self.layer_biases = nn.ParameterList()
        
        self.layer_weights.append(nn.Parameter(torch.empty(D, M, hidden_dim)))
        self.layer_biases.append(nn.Parameter(torch.zeros(D, hidden_dim)))
        
        for _ in range(self.num_layers - 2):
            self.layer_weights.append(nn.Parameter(torch.empty(D, hidden_dim, hidden_dim)))
            self.layer_biases.append(nn.Parameter(torch.zeros(D, hidden_dim)))
        
        self.layer_weights.append(nn.Parameter(torch.empty(D, hidden_dim, 1)))
        self.layer_biases.append(nn.Parameter(torch.zeros(D, 1)))
        
        self._init_weights(init_scale)
    
    def _init_weights(self, init_scale):
        for w, b in zip(self.layer_weights, self.layer_biases):
            std = init_scale * math.sqrt(2.0 / (w.shape[1] + w.shape[2]))
            nn.init.normal_(w, mean=0.0, std=std)
            nn.init.zeros_(b)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, (w, b) in enumerate(zip(self.layer_weights, self.layer_biases)):
            if i == 0:
                x = torch.einsum('bdm,dmh->bdh', x, w)
            else:
                x = torch.einsum('bdh,dho->bdo', x, w)
            x = x + b.unsqueeze(0)
            if i < len(self.layer_weights) - 1:
                x = F.gelu(x)
        return x.squeeze(-1)


class SynapseModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, depth: int = 4, 
                 hidden_mult: float = 2.0, dropout: float = 0.1, init_scale: float = 0.02):
        super().__init__()
        
        dims = [output_dim]
        curr = output_dim
        for _ in range(depth):
            curr = int(curr * hidden_mult)
            dims.append(curr)
        
        self.input_proj = nn.Linear(input_dim, dims[0])
        
        self.encoder = nn.ModuleList()
        for i in range(depth):
            self.encoder.append(nn.Sequential(
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout)
            ))
        
        self.bottleneck = nn.Sequential(
            nn.Linear(dims[depth], dims[depth]),
            nn.LayerNorm(dims[depth]),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.decoder = nn.ModuleList()
        self.skip_proj = nn.ModuleList()
        for i in range(depth):
            self.skip_proj.append(nn.Linear(dims[depth - i] * 2, dims[depth - i - 1]))
            self.decoder.append(nn.Sequential(
                nn.Linear(dims[depth - i - 1], dims[depth - i - 1]),
                nn.LayerNorm(dims[depth - i - 1]),
                nn.GELU(),
                nn.Dropout(dropout)
            ))
        
        self.output_proj = nn.Linear(dims[0], output_dim)
        self._init_weights(init_scale)
    
    def _init_weights(self, scale):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        enc_outs = [x]
        for layer in self.encoder:
            x = layer(x)
            enc_outs.append(x)
        x = self.bottleneck(x)
        for i, (dec, skip) in enumerate(zip(self.decoder, self.skip_proj)):
            x = torch.cat([x, enc_outs[-(i + 1)]], dim=-1)
            x = skip(x)
            x = dec(x)
        return self.output_proj(x)


class SynchronizationModule(nn.Module):
    def __init__(self, D: int, D_sample: int, eps: float = 1e-8):
        super().__init__()
        self.D = D
        self.D_sample = D_sample
        self.eps = eps
        self.decay_rates = nn.Parameter(torch.zeros(D_sample))
        
        idx_i, idx_j = self._gen_pairs(D, D_sample)
        self.register_buffer('idx_i', idx_i)
        self.register_buffer('idx_j', idx_j)
    
    def _gen_pairs(self, D, n):
        i = torch.randint(0, D, (n * 2,))
        j = torch.randint(0, D, (n * 2,))
        mask = i != j
        return i[mask][:n].long(), j[mask][:n].long()
    
    def forward(self, z_hist: torch.Tensor) -> torch.Tensor:
        B, t, D = z_hist.shape
        r = F.softplus(self.decay_rates)
        z_i = z_hist[:, :, self.idx_i]
        z_j = z_hist[:, :, self.idx_j]
        times = torch.arange(t, device=z_hist.device).float()
        decay = torch.exp(-r.unsqueeze(0) * (t - 1 - times).unsqueeze(1))
        num = (z_i * z_j * decay.unsqueeze(0)).sum(dim=1)
        den = torch.sqrt(decay.sum(dim=0) + self.eps)
        return num / den.unsqueeze(0)


class DualSynchronization(nn.Module):
    def __init__(self, D: int, D_action: int, D_output: int, eps: float = 1e-8):
        super().__init__()
        self.action = SynchronizationModule(D, D_action, eps)
        self.output = SynchronizationModule(D, D_output, eps)
    
    def forward(self, z_hist):
        return self.action(z_hist), self.output(z_hist)


class ContinuousThoughtMachine(nn.Module):
    def __init__(self, config: CTMConfig):
        super().__init__()
        self.config = config
        self.D = config.D
        self.M = config.M
        self.T_max = config.T_max
        self.d_embed = config.d_embed
        
        self.nlms = NeuronLevelModels(config.D, config.M, config.nlm_hidden_dim, 
                                       config.nlm_num_layers, config.init_scale)
        
        self.synapse = SynapseModel(config.D + config.d_embed, config.D, config.synapse_depth,
                                     config.synapse_hidden_mult, config.synapse_dropout, config.init_scale)
        
        self.sync = DualSynchronization(config.D, config.D_action, config.D_output, config.sync_eps)
        
        self.attn = nn.MultiheadAttention(config.d_embed, config.num_heads, 
                                           config.attention_dropout, batch_first=True)
        
        self.q_proj = nn.Sequential(nn.Linear(config.D_action, config.d_embed), nn.LayerNorm(config.d_embed))
        
        self.out_proj = nn.Sequential(
            nn.Linear(config.D_output, config.D_output // 2),
            nn.GELU(),
            nn.LayerNorm(config.D_output // 2),
            nn.Linear(config.D_output // 2, config.num_classes)
        )
        
        self.z_init = nn.Parameter(torch.randn(1, config.D) * config.init_scale)
        self.hist_init = nn.Parameter(torch.randn(1, config.D, config.M) * config.init_scale)
    
    def _certainty(self, p):
        eps = 1e-8
        if p.dim() == 3 and p.shape[-1] == 1:
            p = p.squeeze(-1)
        if p.dim() == 2:
            ent = -(p * torch.log2(p + eps) + (1 - p) * torch.log2(1 - p + eps))
            return 1 - ent
        ent = -(p * torch.log2(p + eps)).sum(-1)
        return 1 - ent / math.log2(p.shape[-1])
    
    def forward(self, ctx: torch.Tensor, mask: Optional[torch.Tensor] = None,
                T: Optional[int] = None, return_all: bool = False) -> Dict[str, torch.Tensor]:
        T = T or self.T_max
        B = ctx.shape[0]
        dev = ctx.device
        
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(1)
        
        z = self.z_init.expand(B, -1).clone()
        hist = self.hist_init.expand(B, -1, -1).clone()
        z_hist = z.unsqueeze(1)
        
        preds, attns = [], []
        
        for _ in range(T):
            s_act, s_out = self.sync(z_hist)
            q = self.q_proj(s_act).unsqueeze(1)
            o, w = self.attn(q, ctx, ctx, key_padding_mask=mask, need_weights=True)
            o = o.squeeze(1)
            attns.append(w.squeeze(1))
            
            a = self.synapse(torch.cat([z, o], -1))
            hist = torch.cat([hist[:, :, 1:], a.unsqueeze(-1)], -1)
            z = self.nlms(hist)
            z_hist = torch.cat([z_hist, z.unsqueeze(1)], 1)
            
            pred = self.out_proj(s_out)
            pred = torch.sigmoid(pred) if self.config.num_classes == 1 else F.softmax(pred, -1)
            preds.append(pred)
        
        preds = torch.stack(preds, 1)
        certs = self._certainty(preds)
        attns = torch.stack(attns, 1)
        
        idx = certs.argmax(1)
        batch_idx = torch.arange(B, device=dev)
        final = preds[batch_idx, idx]
        
        out = {'predictions': preds, 'certainties': certs, 'final_prediction': final, 'attention_weights': attns}
        if return_all:
            out['z_history'] = z_hist
        return out
    
    def get_prediction_at_tick(self, out, tick):
        return out['predictions'][:, tick]
    
    def get_most_certain_prediction(self, out):
        c = out['certainties']
        idx = c.argmax(1)
        b = torch.arange(c.shape[0], device=c.device)
        return out['predictions'][b, idx], c[b, idx]


class CTMLoss(nn.Module):
    def __init__(self, num_classes: int = 1, aux_weight: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.aux_weight = aux_weight
    
    def forward(self, preds, certs, targets, z_history=None, return_components=False):
        B, T = preds.shape[:2]
        dev = preds.device
        
        if self.num_classes == 1:
            p = preds.squeeze(-1) if preds.dim() == 3 else preds
            tgt = targets.unsqueeze(1).expand(-1, T)
            loss_t = F.binary_cross_entropy(p, tgt.float(), reduction='none')
        else:
            tgt = targets.unsqueeze(1).expand(-1, T)
            loss_t = F.cross_entropy(preds.view(-1, self.num_classes), tgt.reshape(-1), reduction='none').view(B, T)
        
        _, t1 = loss_t.min(1)
        _, t2 = certs.max(1)
        b = torch.arange(B, device=dev)
        
        loss = (loss_t[b, t1] + loss_t[b, t2]) / 2
        if self.aux_weight > 0:
            loss = loss + self.aux_weight * loss_t.mean(1)
        
        final = loss.mean()
        
        if return_components:
            return final, {
                'primary_loss': loss.mean(),
                'loss_at_t1': loss_t[b, t1].mean(),
                'loss_at_t2': loss_t[b, t2].mean(),
                'avg_t1': t1.float().mean(),
                'avg_t2': t2.float().mean(),
                'avg_certainty': certs.mean(),
                'max_certainty': certs.max(1)[0].mean()
            }
        return final


class CTMTrainer:
    def __init__(self, model, loss_fn, optimizer, scheduler=None, device="cuda", gradient_clip=1.0):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.gradient_clip = gradient_clip
        self.epoch = 0
    
    def train_epoch(self, loader, log_interval=100):
        self.model.train()
        total_loss, total_n = 0.0, 0
        metrics = []
        
        for batch in tqdm(loader, desc=f"Epoch {self.epoch}"):
            ctx = batch['context'].to(self.device)
            tgt = batch['targets'].to(self.device)
            
            self.optimizer.zero_grad()
            out = self.model(ctx, return_all=True)
            loss, comp = self.loss_fn(out['predictions'], out['certainties'], tgt, 
                                       out.get('z_history'), return_components=True)
            loss.backward()
            
            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            
            total_loss += loss.item() * ctx.shape[0]
            total_n += ctx.shape[0]
            metrics.append({k: v.item() if torch.is_tensor(v) else v for k, v in comp.items()})
        
        self.epoch += 1
        return {
            'loss': total_loss / total_n,
            'avg_certainty': sum(m['avg_certainty'] for m in metrics) / len(metrics),
            'avg_t1': sum(m['avg_t1'] for m in metrics) / len(metrics),
            'avg_t2': sum(m['avg_t2'] for m in metrics) / len(metrics)
        }
    
    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss, correct, total_n = 0.0, 0, 0
        
        for batch in tqdm(loader, desc="Eval"):
            ctx = batch['context'].to(self.device)
            tgt = batch['targets'].to(self.device)
            out = self.model(ctx)
            loss = self.loss_fn(out['predictions'], out['certainties'], tgt)
            
            total_loss += loss.item() * ctx.shape[0]
            total_n += ctx.shape[0]
            
            pred = out['final_prediction']
            pred = (pred.squeeze(-1) > 0.5).long() if pred.shape[-1] == 1 else pred.argmax(-1)
            correct += (pred == tgt).sum().item()
        
        return {'loss': total_loss / total_n, 'accuracy': correct / total_n}
    
    def save_checkpoint(self, path):
        torch.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 
                    'epoch': self.epoch}, path)
    
    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.epoch = ckpt['epoch']


def create_optimizer(model, lr=1e-4, weight_decay=0.01):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if 'bias' in n or 'norm' in n else decay).append(p)
    return torch.optim.AdamW([{'params': decay, 'weight_decay': weight_decay},
                               {'params': no_decay, 'weight_decay': 0.0}], lr=lr)


def create_scheduler(optimizer, num_steps, warmup=0.1):
    warmup_steps = int(num_steps * warmup)
    def lr_fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, num_steps - warmup_steps)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * prog)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)