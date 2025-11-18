#!/usr/bin/env python3
"""Dataset para StateNet."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch import LongTensor


class PrefixDataset(Dataset):
    """Dataset de prefijos para entrenar StateEncoder."""

    def __init__(
        self,
        csv_path: Path,
        vocab: dict[str, int],
        symbol_to_idx: dict[str, int],
        max_len: int = 64,
    ):
        """
        Args:
            csv_path: Ruta al CSV con prefijos
            vocab: Mapeo char -> id (incluye <PAD>, <EPS>, A-L)
            symbol_to_idx: Mapeo next_symbol -> índice para clasificación
            max_len: Longitud máxima de secuencia
        """
        self.df = pd.read_csv(csv_path)
        self.vocab = vocab
        self.symbol_to_idx = symbol_to_idx
        self.max_len = max_len

        # Verificar que tenemos las columnas necesarias
        required_cols = {"dfa_id", "prefix_ids", "length", "next_symbol", "final_label"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Columnas faltantes en CSV: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # Parsear prefix_ids (está como JSON string)
        prefix_ids_list = json.loads(row["prefix_ids"])
        length = int(row["length"])

        # Convertir a tensor y pad/truncar
        prefix_ids = torch.tensor(prefix_ids_list[: self.max_len], dtype=torch.long)
        if len(prefix_ids) < self.max_len:
            pad_id = self.vocab["<PAD>"]
            padding = torch.full((self.max_len - len(prefix_ids),), pad_id, dtype=torch.long)
            prefix_ids = torch.cat([prefix_ids, padding])

        # next_symbol -> índice
        next_symbol = str(row["next_symbol"])
        if next_symbol not in self.symbol_to_idx:
            # Si <EOS> no está en el mapeo, usar el último índice
            next_symbol_idx = len(self.symbol_to_idx) - 1
        else:
            next_symbol_idx = self.symbol_to_idx[next_symbol]

        # final_label
        final_label = float(row["final_label"])

        # dfa_id
        dfa_id = int(row["dfa_id"])

        return {
            "prefix_ids": prefix_ids,  # [max_len]
            "length": torch.tensor(length, dtype=torch.long),  # scalar
            "next_symbol": torch.tensor(next_symbol_idx, dtype=torch.long),  # scalar
            "final_label": torch.tensor(final_label, dtype=torch.float),  # scalar
            "dfa_id": torch.tensor(dfa_id, dtype=torch.long),  # scalar
        }


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate function para DataLoader."""
    return {
        "prefix_ids": torch.stack([item["prefix_ids"] for item in batch]),  # [batch, max_len]
        "lengths": torch.stack([item["length"] for item in batch]),  # [batch]
        "next_symbol": torch.stack([item["next_symbol"] for item in batch]),  # [batch]
        "final_label": torch.stack([item["final_label"] for item in batch]),  # [batch]
        "dfa_ids": torch.stack([item["dfa_id"] for item in batch]),  # [batch]
    }

