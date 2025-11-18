#!/usr/bin/env python3
"""Script de entrenamiento para StateEncoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset import PrefixDataset, collate_fn
from src.model import StateEncoder, StateEncoderWithHeads


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Entrenar StateEncoder")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root / "data" / "statenet",
        help="Directorio con prefix_train.csv y prefix_val.csv",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=root / "vocab_char_to_id.json",
        help="Ruta al vocabulario",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "checkpoints",
        help="Directorio para guardar checkpoints",
    )
    parser.add_argument(
        "--emb-dim",
        type=int,
        default=128,
        help="Dimensión de embedding",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Dimensión oculta del RNN",
    )
    parser.add_argument(
        "--d-state",
        type=int,
        default=128,
        help="Dimensión del estado codificado",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help="Número de capas RNN",
    )
    parser.add_argument(
        "--rnn-type",
        type=str,
        default="GRU",
        choices=["GRU", "LSTM"],
        help="Tipo de RNN",
    )
    parser.add_argument(
        "--use-dfa-embedding",
        action="store_true",
        help="Usar embedding de dfa_id",
    )
    parser.add_argument(
        "--dfa-emb-dim",
        type=int,
        default=16,
        help="Dimensión del embedding de DFA",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Tamaño de batch",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--lambda-label",
        type=float,
        default=1.0,
        help="Peso λ para loss de final_label",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=50,
        help="Máximo número de épocas",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Paciencia para early stopping",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Dispositivo (cuda/cpu)",
    )
    return parser.parse_args()


def load_vocab(vocab_path: Path) -> dict[str, int]:
    with vocab_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_symbol_to_idx(vocab: dict[str, int]) -> dict[str, int]:
    """
    Construye mapeo de símbolos (A-L + <EOS>) a índices para clasificación.
    
    Asume que <EOS> no está en vocab, así que lo agregamos al final.
    """
    symbols = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "<EOS>"]
    symbol_to_idx = {}
    for idx, sym in enumerate(symbols):
        symbol_to_idx[sym] = idx
    return symbol_to_idx


def compute_loss(
    next_symbol_logits: torch.Tensor,
    next_symbol_target: torch.Tensor,
    final_label_logits: torch.Tensor,
    final_label_target: torch.Tensor,
    lambda_label: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Calcula loss combinado.

    Returns:
        total_loss: Loss total
        metrics: Diccionario con métricas
    """
    # Task 1: next_symbol (CrossEntropy)
    ce_loss = nn.functional.cross_entropy(next_symbol_logits, next_symbol_target)
    
    # Task 2: final_label (BCE)
    bce_loss = nn.functional.binary_cross_entropy_with_logits(
        final_label_logits.squeeze(-1), final_label_target
    )

    total_loss = ce_loss + lambda_label * bce_loss

    # Métricas
    with torch.no_grad():
        next_symbol_pred = next_symbol_logits.argmax(dim=-1)
        next_symbol_acc = (next_symbol_pred == next_symbol_target).float().mean().item()

        final_label_pred = (torch.sigmoid(final_label_logits.squeeze(-1)) > 0.5).float()
        final_label_acc = (final_label_pred == final_label_target).float().mean().item()

    metrics = {
        "ce_loss": ce_loss.item(),
        "bce_loss": bce_loss.item(),
        "total_loss": total_loss.item(),
        "next_symbol_acc": next_symbol_acc,
        "final_label_acc": final_label_acc,
    }

    return total_loss, metrics


def train_epoch(
    model: StateEncoderWithHeads,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    lambda_label: float,
) -> dict[str, float]:
    model.train()
    total_metrics = {"ce_loss": 0.0, "bce_loss": 0.0, "total_loss": 0.0, "next_symbol_acc": 0.0, "final_label_acc": 0.0}
    n_batches = 0

    for batch in dataloader:
        prefix_ids = batch["prefix_ids"].to(device)
        lengths = batch["lengths"].to(device)
        next_symbol = batch["next_symbol"].to(device)
        final_label = batch["final_label"].to(device)
        dfa_ids = batch["dfa_ids"].to(device) if "dfa_ids" in batch else None

        optimizer.zero_grad()

        next_symbol_logits, final_label_logits = model(prefix_ids, lengths, dfa_ids)

        loss, metrics = compute_loss(
            next_symbol_logits, next_symbol, final_label_logits, final_label, lambda_label
        )

        loss.backward()
        optimizer.step()

        for k, v in metrics.items():
            total_metrics[k] += v
        n_batches += 1

    return {k: v / n_batches for k, v in total_metrics.items()}


def validate(
    model: StateEncoderWithHeads,
    dataloader: DataLoader,
    device: torch.device,
    lambda_label: float,
) -> dict[str, float]:
    model.eval()
    total_metrics = {"ce_loss": 0.0, "bce_loss": 0.0, "total_loss": 0.0, "next_symbol_acc": 0.0, "final_label_acc": 0.0}
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            prefix_ids = batch["prefix_ids"].to(device)
            lengths = batch["lengths"].to(device)
            next_symbol = batch["next_symbol"].to(device)
            final_label = batch["final_label"].to(device)
            dfa_ids = batch["dfa_ids"].to(device) if "dfa_ids" in batch else None

            next_symbol_logits, final_label_logits = model(prefix_ids, lengths, dfa_ids)

            _, metrics = compute_loss(
                next_symbol_logits, next_symbol, final_label_logits, final_label, lambda_label
            )

            for k, v in metrics.items():
                total_metrics[k] += v
            n_batches += 1

    return {k: v / n_batches for k, v in total_metrics.items()}


def main() -> None:
    args = parse_args()

    # Cargar vocabulario
    vocab = load_vocab(args.vocab)
    vocab_size = len(vocab)
    symbol_to_idx = build_symbol_to_idx(vocab)

    # Cargar meta para obtener max_len
    meta_path = args.data_dir / "meta.json"
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    max_len = meta["max_len"]

    # Obtener número de DFAs si se usa embedding
    if args.use_dfa_embedding:
        train_df = pd.read_csv(args.data_dir / "prefix_train.csv", usecols=["dfa_id"])
        num_dfas = int(train_df["dfa_id"].max() + 1)
    else:
        num_dfas = None

    # Datasets
    train_dataset = PrefixDataset(
        args.data_dir / "prefix_train.csv", vocab, symbol_to_idx, max_len
    )
    val_dataset = PrefixDataset(
        args.data_dir / "prefix_val.csv", vocab, symbol_to_idx, max_len
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Modelo
    encoder = StateEncoder(
        vocab_size=vocab_size,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        d_state=args.d_state,
        max_len=max_len,
        num_layers=args.num_layers,
        rnn_type=args.rnn_type,
        use_dfa_embedding=args.use_dfa_embedding,
        num_dfas=num_dfas,
        dfa_emb_dim=args.dfa_emb_dim,
        padding_idx=vocab["<PAD>"],
    )

    model = StateEncoderWithHeads(encoder, vocab_size, num_symbols=len(symbol_to_idx))
    model = model.to(args.device)

    # Optimizador
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Entrenamiento
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    train_log = []

    print(f"Entrenando en {args.device}")
    print(f"Vocab size: {vocab_size}, Max len: {max_len}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    for epoch in range(1, args.max_epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, args.device, args.lambda_label)
        val_metrics = validate(model, val_loader, args.device, args.lambda_label)

        print(
            f"Epoch {epoch}/{args.max_epochs} | "
            f"Train Loss: {train_metrics['total_loss']:.4f} "
            f"(CE: {train_metrics['ce_loss']:.4f}, BCE: {train_metrics['bce_loss']:.4f}) | "
            f"Train Acc: next={train_metrics['next_symbol_acc']:.4f}, label={train_metrics['final_label_acc']:.4f} | "
            f"Val Loss: {val_metrics['total_loss']:.4f} | "
            f"Val Acc: next={val_metrics['next_symbol_acc']:.4f}, label={val_metrics['final_label_acc']:.4f}"
        )

        train_log.append({"epoch": epoch, **train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}})

        # Early stopping
        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            patience_counter = 0

            # Guardar mejor modelo
            torch.save(encoder.state_dict(), output_dir / "state_encoder.pth")

            # Guardar hparams
            hparams = {
                "vocab_size": vocab_size,
                "emb_dim": args.emb_dim,
                "hidden_dim": args.hidden_dim,
                "d_state": args.d_state,
                "max_len": max_len,
                "num_layers": args.num_layers,
                "rnn_type": args.rnn_type,
                "use_dfa_embedding": args.use_dfa_embedding,
                "dfa_emb_dim": args.dfa_emb_dim if args.use_dfa_embedding else None,
                "num_dfas": num_dfas if args.use_dfa_embedding else None,
                "padding_idx": vocab["<PAD>"],
            }
            with (output_dir / "statenet_hparams.json").open("w", encoding="utf-8") as f:
                json.dump(hparams, f, indent=2)

            print(f"  ✓ Modelo guardado (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping en epoch {epoch}")
                break

    # Guardar log
    log_df = pd.DataFrame(train_log)
    log_df.to_csv(output_dir / "train_log.csv", index=False)

    print(f"\nEntrenamiento completado. Mejor val_loss: {best_val_loss:.4f}")
    print(f"Modelo guardado en: {output_dir / 'state_encoder.pth'}")


if __name__ == "__main__":
    main()

