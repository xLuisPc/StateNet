#!/usr/bin/env python3
"""
StateNet - Entrenamiento completo para Colab
Script standalone que incluye todo lo necesario para entrenar.
"""

import json
import os
from pathlib import Path
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

CONFIG = {
    # Rutas (ajustar según tu estructura en Colab)
    "vocab_path": "/content/StateNet/vocab_char_to_id.json",
    "data_dir": "/content/StateNet/data/statenet",
    "output_dir": "/content/StateNet/checkpoints",
    
    # Hiperparámetros del modelo
    "emb_dim": 128,
    "hidden_dim": 256,
    "d_state": 128,
    "num_layers": 1,
    "rnn_type": "GRU",  # "GRU" o "LSTM"
    "use_dfa_embedding": False,
    "dfa_emb_dim": 16,
    
    # Hiperparámetros de entrenamiento
    "batch_size": 64,
    "lr": 1e-3,
    "lambda_label": 1.0,
    "max_epochs": 50,
    "patience": 5,
    
    # Dispositivo (se detecta automáticamente)
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# ============================================================================
# MODELO STATEENCODER
# ============================================================================

class StateEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, d_state, max_len=64, 
                 num_layers=1, rnn_type="GRU", use_dfa_embedding=False, 
                 num_dfas=None, dfa_emb_dim=16, padding_idx=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.d_state = d_state
        self.max_len = max_len
        self.num_layers = num_layers
        self.rnn_type = rnn_type.upper()
        self.use_dfa_embedding = use_dfa_embedding
        self.padding_idx = padding_idx

        self.token_embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        
        if self.rnn_type == "GRU":
            self.rnn = nn.GRU(emb_dim, hidden_dim, num_layers=num_layers, 
                             batch_first=True, bidirectional=False)
        elif self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, 
                              batch_first=True, bidirectional=False)
        else:
            raise ValueError(f"rnn_type debe ser 'GRU' o 'LSTM', recibido: {rnn_type}")

        if use_dfa_embedding:
            if num_dfas is None:
                raise ValueError("num_dfas requerido si use_dfa_embedding=True")
            self.dfa_embedding = nn.Embedding(num_dfas, dfa_emb_dim)
            rnn_output_dim = hidden_dim
        else:
            self.dfa_embedding = None
            rnn_output_dim = hidden_dim

        self.state_proj = nn.Linear(rnn_output_dim + (dfa_emb_dim if use_dfa_embedding else 0), d_state)

    def forward(self, prefix_ids, lengths, dfa_ids=None):
        batch_size = prefix_ids.size(0)
        embedded = self.token_embedding(prefix_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        if self.rnn_type == "GRU":
            packed_output, hidden = self.rnn(packed)
        else:
            packed_output, (hidden, _) = self.rnn(packed)

        if self.num_layers == 1:
            h_rnn = hidden.squeeze(0)
        else:
            h_rnn = hidden[-1]

        if self.use_dfa_embedding:
            if dfa_ids is None:
                raise ValueError("dfa_ids requerido cuando use_dfa_embedding=True")
            dfa_emb = self.dfa_embedding(dfa_ids)
            h_concat = torch.cat([h_rnn, dfa_emb], dim=1)
        else:
            h_concat = h_rnn

        h_t = self.state_proj(h_concat)
        return h_t


class StateEncoderWithHeads(nn.Module):
    def __init__(self, encoder, vocab_size, num_symbols=13):
        super().__init__()
        self.encoder = encoder
        self.vocab_size = vocab_size
        self.num_symbols = num_symbols
        self.next_symbol_head = nn.Linear(encoder.d_state, num_symbols)
        self.final_label_head = nn.Linear(encoder.d_state, 1)

    def forward(self, prefix_ids, lengths, dfa_ids=None):
        h_t = self.encoder(prefix_ids, lengths, dfa_ids)
        next_symbol_logits = self.next_symbol_head(h_t)
        final_label_logits = self.final_label_head(h_t)
        return next_symbol_logits, final_label_logits


# ============================================================================
# DATASET
# ============================================================================

class PrefixDataset(Dataset):
    def __init__(self, csv_path, vocab, symbol_to_idx, max_len=64):
        # Leer CSV asegurando que prefix_ids se lea como string
        self.df = pd.read_csv(csv_path, dtype={'prefix_ids': str}, keep_default_na=False, na_values=[''])
        # Reemplazar strings vacíos con NaN para manejo consistente
        self.df['prefix_ids'] = self.df['prefix_ids'].replace('', pd.NA)
        self.vocab = vocab
        self.symbol_to_idx = symbol_to_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Manejar prefix_ids (puede ser string JSON o ya una lista)
        prefix_ids_raw = row["prefix_ids"]
        if pd.isna(prefix_ids_raw):
            prefix_ids_list = []
        elif isinstance(prefix_ids_raw, str):
            try:
                prefix_ids_list = json.loads(prefix_ids_raw)
            except (json.JSONDecodeError, ValueError):
                prefix_ids_list = []
        elif isinstance(prefix_ids_raw, (list, tuple)):
            prefix_ids_list = list(prefix_ids_raw)
        else:
            # Si es float u otro tipo, intentar convertir
            try:
                prefix_ids_list = json.loads(str(prefix_ids_raw))
            except:
                prefix_ids_list = []
        
        # Asegurar que prefix_ids_list sea una lista válida
        if not isinstance(prefix_ids_list, list):
            prefix_ids_list = []
        
        # Calcular length basado en los IDs reales (sin padding)
        actual_length = len(prefix_ids_list)
        if actual_length == 0:
            # Si está vacío, usar al menos 1 (solo padding, pero length=1)
            prefix_ids_list = [self.vocab.get("<PAD>", 0)]
            actual_length = 1
        
        # Truncar si es necesario
        prefix_ids_list = prefix_ids_list[:self.max_len]
        actual_length = min(actual_length, self.max_len)
        
        # Usar el length del CSV si está disponible y es válido, sino usar el calculado
        csv_length = int(row["length"]) if not pd.isna(row["length"]) and int(row["length"]) > 0 else None
        length = csv_length if csv_length is not None else actual_length
        length = max(1, min(length, self.max_len))  # Asegurar 1 <= length <= max_len

        prefix_ids = torch.tensor(prefix_ids_list[:self.max_len], dtype=torch.long)
        if len(prefix_ids) < self.max_len:
            pad_id = self.vocab["<PAD>"]
            padding = torch.full((self.max_len - len(prefix_ids),), pad_id, dtype=torch.long)
            prefix_ids = torch.cat([prefix_ids, padding])

        next_symbol = str(row["next_symbol"]) if not pd.isna(row["next_symbol"]) else "<EOS>"
        if next_symbol not in self.symbol_to_idx:
            next_symbol_idx = len(self.symbol_to_idx) - 1
        else:
            next_symbol_idx = self.symbol_to_idx[next_symbol]

        final_label = float(row["final_label"]) if not pd.isna(row["final_label"]) else 0.0
        dfa_id = int(row["dfa_id"]) if not pd.isna(row["dfa_id"]) else 0

        return {
            "prefix_ids": prefix_ids,
            "length": torch.tensor(length, dtype=torch.long),
            "next_symbol": torch.tensor(next_symbol_idx, dtype=torch.long),
            "final_label": torch.tensor(final_label, dtype=torch.float),
            "dfa_id": torch.tensor(dfa_id, dtype=torch.long),
        }


def collate_fn(batch):
    return {
        "prefix_ids": torch.stack([item["prefix_ids"] for item in batch]),
        "lengths": torch.stack([item["length"] for item in batch]),
        "next_symbol": torch.stack([item["next_symbol"] for item in batch]),
        "final_label": torch.stack([item["final_label"] for item in batch]),
        "dfa_ids": torch.stack([item["dfa_id"] for item in batch]),
    }


# ============================================================================
# FUNCIONES DE ENTRENAMIENTO
# ============================================================================

def compute_loss(next_symbol_logits, next_symbol_target, final_label_logits, 
                 final_label_target, lambda_label):
    ce_loss = nn.functional.cross_entropy(next_symbol_logits, next_symbol_target)
    bce_loss = nn.functional.binary_cross_entropy_with_logits(
        final_label_logits.squeeze(-1), final_label_target
    )
    total_loss = ce_loss + lambda_label * bce_loss
    
    with torch.no_grad():
        next_symbol_pred = next_symbol_logits.argmax(dim=-1)
        next_symbol_acc = (next_symbol_pred == next_symbol_target).float().mean().item()
        final_label_pred = (torch.sigmoid(final_label_logits.squeeze(-1)) > 0.5).float()
        final_label_acc = (final_label_pred == final_label_target).float().mean().item()
    
    return total_loss, {
        'ce_loss': ce_loss.item(),
        'bce_loss': bce_loss.item(),
        'total_loss': total_loss.item(),
        'next_symbol_acc': next_symbol_acc,
        'final_label_acc': final_label_acc,
    }


def train_epoch(model, dataloader, optimizer, device, lambda_label):
    model.train()
    total_metrics = {'ce_loss': 0.0, 'bce_loss': 0.0, 'total_loss': 0.0, 
                     'next_symbol_acc': 0.0, 'final_label_acc': 0.0}
    n_batches = 0
    
    for batch in dataloader:
        prefix_ids = batch['prefix_ids'].to(device)
        lengths = batch['lengths'].to(device)
        next_symbol = batch['next_symbol'].to(device)
        final_label = batch['final_label'].to(device)
        dfa_ids = batch.get('dfa_ids')
        dfa_ids = dfa_ids.to(device) if dfa_ids is not None else None
        
        optimizer.zero_grad()
        next_symbol_logits, final_label_logits = model(prefix_ids, lengths, dfa_ids)
        loss, metrics = compute_loss(
            next_symbol_logits, next_symbol, final_label_logits, 
            final_label, lambda_label
        )
        loss.backward()
        optimizer.step()
        
        for k, v in metrics.items():
            total_metrics[k] += v
        n_batches += 1
    
    return {k: v / n_batches for k, v in total_metrics.items()}


def validate(model, dataloader, device, lambda_label):
    model.eval()
    total_metrics = {'ce_loss': 0.0, 'bce_loss': 0.0, 'total_loss': 0.0, 
                     'next_symbol_acc': 0.0, 'final_label_acc': 0.0}
    n_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            prefix_ids = batch['prefix_ids'].to(device)
            lengths = batch['lengths'].to(device)
            next_symbol = batch['next_symbol'].to(device)
            final_label = batch['final_label'].to(device)
            dfa_ids = batch.get('dfa_ids')
            dfa_ids = dfa_ids.to(device) if dfa_ids is not None else None
            
            next_symbol_logits, final_label_logits = model(prefix_ids, lengths, dfa_ids)
            _, metrics = compute_loss(
                next_symbol_logits, next_symbol, final_label_logits, 
                final_label, lambda_label
            )
            
            for k, v in metrics.items():
                total_metrics[k] += v
            n_batches += 1
    
    return {k: v / n_batches for k, v in total_metrics.items()}


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def main():
    config = CONFIG.copy()
    
    print("=" * 60)
    print("🚀 StateNet - Entrenamiento en Colab")
    print("=" * 60)
    
    # Verificar GPU
    device = torch.device(config["device"])
    print(f"\n📱 Dispositivo: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memoria: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Verificar archivos
    print("\n📋 Verificando archivos...")
    required_files = [
        config["vocab_path"],
        f"{config['data_dir']}/prefix_train.csv",
        f"{config['data_dir']}/prefix_val.csv",
        f"{config['data_dir']}/meta.json"
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✓ {file}")
        else:
            print(f"   ✗ FALTA: {file}")
            raise FileNotFoundError(f"Falta el archivo: {file}")
    
    # Cargar vocabulario
    print("\n📚 Cargando vocabulario...")
    with open(config["vocab_path"], 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    print(f"   Vocab size: {vocab_size}")
    
    # Construir symbol_to_idx
    symbols = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "<EOS>"]
    symbol_to_idx = {sym: idx for idx, sym in enumerate(symbols)}
    
    # Cargar meta
    with open(f"{config['data_dir']}/meta.json", 'r') as f:
        meta = json.load(f)
    max_len = meta['max_len']
    print(f"   Max len: {max_len}")
    
    # Crear datasets
    print("\n📊 Cargando datasets...")
    train_dataset = PrefixDataset(
        f"{config['data_dir']}/prefix_train.csv",
        vocab, symbol_to_idx, max_len
    )
    val_dataset = PrefixDataset(
        f"{config['data_dir']}/prefix_val.csv",
        vocab, symbol_to_idx, max_len
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'],
        shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'],
        shuffle=False, collate_fn=collate_fn
    )
    
    print(f"   Train samples: {len(train_dataset):,}")
    print(f"   Val samples: {len(val_dataset):,}")
    
    # Crear modelo
    print("\n🧠 Creando modelo...")
    encoder = StateEncoder(
        vocab_size=vocab_size,
        emb_dim=config['emb_dim'],
        hidden_dim=config['hidden_dim'],
        d_state=config['d_state'],
        max_len=max_len,
        num_layers=config['num_layers'],
        rnn_type=config['rnn_type'],
        use_dfa_embedding=config['use_dfa_embedding'],
        num_dfas=None,  # Se calcularía si use_dfa_embedding=True
        dfa_emb_dim=config['dfa_emb_dim'],
        padding_idx=vocab['<PAD>'],
    )
    
    model = StateEncoderWithHeads(encoder, vocab_size, num_symbols=len(symbol_to_idx))
    model = model.to(device)
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parámetros totales: {total_params:,}")
    print(f"   Parámetros entrenables: {trainable_params:,}")
    
    # Optimizador
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    # Crear directorio de salida
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
    
    # Entrenamiento
    print("\n🎯 Iniciando entrenamiento...")
    print("=" * 60)
    
    # Verificar uso de GPU RAM
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print(f"\n💾 Estado inicial GPU:")
        print(f"   RAM usada: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"   RAM reservada: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_log = []
    
    for epoch in range(1, config['max_epochs'] + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device, config['lambda_label'])
        val_metrics = validate(model, val_loader, device, config['lambda_label'])
        
        # Mostrar métricas y uso de GPU
        gpu_info = ""
        if torch.cuda.is_available():
            gpu_used = torch.cuda.memory_allocated(0) / 1e9
            gpu_reserved = torch.cuda.memory_reserved(0) / 1e9
            gpu_info = f" | GPU: {gpu_used:.2f}/{gpu_reserved:.2f} GB"
        
        print(
            f"Epoch {epoch:3d}/{config['max_epochs']} | "
            f"Train Loss: {train_metrics['total_loss']:.4f} "
            f"(CE: {train_metrics['ce_loss']:.4f}, BCE: {train_metrics['bce_loss']:.4f}) | "
            f"Train Acc: next={train_metrics['next_symbol_acc']:.4f}, label={train_metrics['final_label_acc']:.4f} | "
            f"Val Loss: {val_metrics['total_loss']:.4f} | "
            f"Val Acc: next={val_metrics['next_symbol_acc']:.4f}, label={val_metrics['final_label_acc']:.4f}"
            + gpu_info
        )
        
        train_log.append({
            'epoch': epoch,
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()}
        })
        
        # Early stopping y guardado
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            patience_counter = 0
            
            # Guardar modelo
            torch.save(encoder.state_dict(), f"{config['output_dir']}/state_encoder.pth")
            
            # Guardar hparams
            hparams = {
                'vocab_size': vocab_size,
                'emb_dim': config['emb_dim'],
                'hidden_dim': config['hidden_dim'],
                'd_state': config['d_state'],
                'max_len': max_len,
                'num_layers': config['num_layers'],
                'rnn_type': config['rnn_type'],
                'use_dfa_embedding': config['use_dfa_embedding'],
                'dfa_emb_dim': config['dfa_emb_dim'] if config['use_dfa_embedding'] else None,
                'num_dfas': None,
                'padding_idx': vocab['<PAD>'],
            }
            with open(f"{config['output_dir']}/statenet_hparams.json", 'w') as f:
                json.dump(hparams, f, indent=2)
            
            print(f"  ✓ Modelo guardado (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"\n⏹️  Early stopping en epoch {epoch}")
                break
    
    # Guardar log
    log_df = pd.DataFrame(train_log)
    log_df.to_csv(f"{config['output_dir']}/train_log.csv", index=False)
    
    print("\n" + "=" * 60)
    print("✅ Entrenamiento completado!")
    print(f"   Mejor val_loss: {best_val_loss:.4f}")
    print(f"   Modelo guardado en: {config['output_dir']}/state_encoder.pth")
    
    # Mostrar uso final de GPU
    if torch.cuda.is_available():
        peak_used = torch.cuda.max_memory_allocated(0) / 1e9
        peak_reserved = torch.cuda.max_memory_reserved(0) / 1e9
        print(f"\n💾 Uso máximo de GPU durante entrenamiento:")
        print(f"   RAM usada (pico): {peak_used:.2f} GB")
        print(f"   RAM reservada (pico): {peak_reserved:.2f} GB")
    
    print("=" * 60)


if __name__ == "__main__":
    main()

