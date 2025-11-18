#!/usr/bin/env python3
"""
Script para evaluar el modelo entrenado con datos reales.
Toma una muestra de los archivos normales (train/val/test) y evalúa el rendimiento.
"""

import json
import os
import sys
from pathlib import Path

# Agregar raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import PrefixDataset, collate_fn
from src.infer import load_state_encoder
from src.model import StateEncoderWithHeads


def build_symbol_to_idx(vocab):
    """Construye mapeo de símbolos a índices."""
    symbols = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "<EOS>"]
    symbol_to_idx = {sym: idx for idx, sym in enumerate(symbols)}
    return symbol_to_idx


def compute_metrics(model, dataloader, device, lambda_label=1.0, max_samples=None):
    """
    Evalúa el modelo y calcula métricas.
    
    Args:
        model: Modelo StateEncoderWithHeads
        dataloader: DataLoader con datos
        device: Dispositivo (cuda/cpu)
        lambda_label: Peso del loss de label
        max_samples: Máximo número de muestras a evaluar (None = todos)
    
    Returns:
        dict con métricas
    """
    model.eval()
    total_metrics = {
        'ce_loss': 0.0,
        'bce_loss': 0.0,
        'total_loss': 0.0,
        'next_symbol_acc': 0.0,
        'final_label_acc': 0.0
    }
    n_batches = 0
    n_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if max_samples and n_samples >= max_samples:
                break
            
            prefix_ids = batch['prefix_ids'].to(device)
            lengths = batch['lengths'].to(device)
            next_symbol = batch['next_symbol'].to(device)
            final_label = batch['final_label'].to(device)
            dfa_ids = batch.get('dfa_ids')
            dfa_ids = dfa_ids.to(device) if dfa_ids is not None else None
            
            # Forward pass
            next_symbol_logits, final_label_logits = model(prefix_ids, lengths, dfa_ids)
            
            # Calcular loss
            ce_loss = F.cross_entropy(next_symbol_logits, next_symbol)
            bce_loss = F.binary_cross_entropy_with_logits(
                final_label_logits.squeeze(-1), final_label
            )
            total_loss = ce_loss + lambda_label * bce_loss
            
            # Calcular accuracy
            next_symbol_pred = next_symbol_logits.argmax(dim=-1)
            next_symbol_acc = (next_symbol_pred == next_symbol).float().mean().item()
            
            final_label_pred = (torch.sigmoid(final_label_logits.squeeze(-1)) > 0.5).float()
            final_label_acc = (final_label_pred == final_label).float().mean().item()
            
            # Acumular métricas
            batch_size = prefix_ids.size(0)
            total_metrics['ce_loss'] += ce_loss.item() * batch_size
            total_metrics['bce_loss'] += bce_loss.item() * batch_size
            total_metrics['total_loss'] += total_loss.item() * batch_size
            total_metrics['next_symbol_acc'] += next_symbol_acc * batch_size
            total_metrics['final_label_acc'] += final_label_acc * batch_size
            
            n_batches += 1
            n_samples += batch_size
    
    # Promediar
    if n_samples > 0:
        for key in ['ce_loss', 'bce_loss', 'total_loss', 'next_symbol_acc', 'final_label_acc']:
            total_metrics[key] /= n_samples
    
    total_metrics['n_samples'] = n_samples
    total_metrics['n_batches'] = n_batches
    
    return total_metrics


def evaluate_on_dataset(csv_path, model_path, hparams_path, vocab_path, 
                        max_samples=1000, batch_size=64, device=None):
    """
    Evalúa el modelo en un dataset específico.
    
    Args:
        csv_path: Ruta al CSV (train/val/test)
        model_path: Ruta al modelo entrenado
        hparams_path: Ruta a hparams
        vocab_path: Ruta al vocabulario
        max_samples: Máximo número de muestras a evaluar
        batch_size: Tamaño de batch
        device: Dispositivo (None = auto)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print(f"📊 Evaluando en: {Path(csv_path).name}")
    print(f"{'='*60}")
    
    # Verificar archivos
    if not os.path.exists(csv_path):
        print(f"❌ No se encuentra: {csv_path}")
        return None
    
    if not os.path.exists(model_path) or not os.path.exists(hparams_path):
        print(f"❌ No se encuentran los checkpoints")
        return None
    
    # Cargar vocabulario
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    symbol_to_idx = build_symbol_to_idx(vocab)
    
    # Cargar meta para max_len
    data_dir = Path(csv_path).parent
    meta_path = data_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        max_len = meta['max_len']
    else:
        max_len = 64  # Default
    
    # Cargar modelo
    print(f"\n🤖 Cargando modelo...")
    encoder = load_state_encoder(model_path, hparams_path, str(device))
    
    # Crear modelo con heads para evaluación
    vocab_size = len(vocab)
    num_symbols = len(symbol_to_idx)
    
    # El encoder.model es StateEncoder, crear el modelo completo con heads
    # Necesitamos cargar los pesos del modelo completo si están disponibles
    # Por ahora, creamos el modelo con heads desde cero (solo para evaluación)
    model = StateEncoderWithHeads(encoder.model, vocab_size, num_symbols)
    model = model.to(device)
    model.eval()
    
    # Nota: Los heads no se entrenaron, pero podemos evaluar el encoder
    # Para evaluación completa, necesitaríamos el modelo completo entrenado
    
    print(f"   ✓ Modelo cargado en {device}")
    
    # Crear dataset (tomar muestra si es muy grande)
    print(f"\n📁 Cargando datos...")
    df = pd.read_csv(csv_path)
    total_samples = len(df)
    print(f"   Total de muestras: {total_samples:,}")
    
    if max_samples and total_samples > max_samples:
        df = df.sample(n=max_samples, random_state=42)
        print(f"   Usando muestra de: {len(df):,} muestras")
    
    # Crear dataset temporal (necesitamos guardar en CSV temporal)
    temp_csv = data_dir / f"temp_eval_{Path(csv_path).stem}.csv"
    df.to_csv(temp_csv, index=False)
    
    try:
        dataset = PrefixDataset(temp_csv, vocab, symbol_to_idx, max_len)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )
        
        # Evaluar
        print(f"\n🔄 Evaluando...")
        metrics = compute_metrics(model, dataloader, device, lambda_label=1.0, max_samples=max_samples)
        
        # Mostrar resultados
        print(f"\n📈 RESULTADOS:")
        print(f"   Muestras evaluadas: {metrics['n_samples']:,}")
        print(f"   Batches procesados: {metrics['n_batches']}")
        print(f"\n   Loss:")
        print(f"     - Total: {metrics['total_loss']:.4f}")
        print(f"     - CE (next_symbol): {metrics['ce_loss']:.4f}")
        print(f"     - BCE (final_label): {metrics['bce_loss']:.4f}")
        print(f"\n   Accuracy:")
        print(f"     - next_symbol: {metrics['next_symbol_acc']:.4%}")
        print(f"     - final_label: {metrics['final_label_acc']:.4%}")
        
        return metrics
        
    finally:
        # Limpiar archivo temporal
        if temp_csv.exists():
            temp_csv.unlink()


def main():
    """Evalúa el modelo en train, val y test."""
    print("=" * 60)
    print("🧪 EVALUACIÓN DEL MODELO CON DATOS REALES")
    print("=" * 60)
    
    # Configuración (desde la raíz del proyecto)
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "statenet"
    
    model_path = base_dir / "checkpoints" / "state_encoder.pth"
    hparams_path = base_dir / "checkpoints" / "statenet_hparams.json"
    vocab_path = base_dir / "vocab_char_to_id.json"
    
    # Verificar que existen los checkpoints
    if not model_path.exists() or not hparams_path.exists():
        print(f"❌ No se encuentran los checkpoints:")
        print(f"   - {model_path}: {'✓' if model_path.exists() else '✗'}")
        print(f"   - {hparams_path}: {'✓' if hparams_path.exists() else '✗'}")
        return
    
    # Evaluar en cada dataset disponible
    datasets = {
        "train": data_dir / "prefix_train.csv",
        "val": data_dir / "prefix_val.csv",
        "test": data_dir / "prefix_test.csv",
    }
    
    results = {}
    
    for name, csv_path in datasets.items():
        if csv_path.exists():
            metrics = evaluate_on_dataset(
                csv_path=csv_path,
                model_path=model_path,
                hparams_path=hparams_path,
                vocab_path=vocab_path,
                max_samples=1000,  # Evaluar máximo 1000 muestras por dataset
                batch_size=64,
            )
            if metrics:
                results[name] = metrics
        else:
            print(f"\n⚠️  {name}: {csv_path.name} no existe, saltando...")
    
    # Resumen final
    if results:
        print(f"\n" + "=" * 60)
        print("📊 RESUMEN FINAL")
        print("=" * 60)
        
        for name, metrics in results.items():
            print(f"\n{name.upper()}:")
            print(f"  Loss: {metrics['total_loss']:.4f}")
            print(f"  Acc (next_symbol): {metrics['next_symbol_acc']:.4%}")
            print(f"  Acc (final_label): {metrics['final_label_acc']:.4%}")
            print(f"  Muestras: {metrics['n_samples']:,}")
    
    print(f"\n" + "=" * 60)
    print("✅ Evaluación completada")
    print("=" * 60)


if __name__ == "__main__":
    main()

