#!/usr/bin/env python3
"""Función de inferencia reutilizable para StateEncoder."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import LongTensor, Tensor

from src.model import StateEncoder


class StateEncoderInference:
    """
    Wrapper para inferencia con StateEncoder.

    Carga modelo y hparams, proporciona interfaz simple para codificar prefijos.
    """

    def __init__(self, model_path: Path, hparams_path: Path, device: str | None = None):
        """
        Args:
            model_path: Ruta a state_encoder.pth
            hparams_path: Ruta a statenet_hparams.json
            device: Dispositivo (cuda/cpu). Si None, usa cuda si está disponible.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Cargar hparams
        with hparams_path.open("r", encoding="utf-8") as f:
            self.hparams = json.load(f)

        # Construir modelo
        self.model = StateEncoder(
            vocab_size=self.hparams["vocab_size"],
            emb_dim=self.hparams["emb_dim"],
            hidden_dim=self.hparams["hidden_dim"],
            d_state=self.hparams["d_state"],
            max_len=self.hparams["max_len"],
            num_layers=self.hparams["num_layers"],
            rnn_type=self.hparams["rnn_type"],
            use_dfa_embedding=self.hparams.get("use_dfa_embedding", False),
            num_dfas=self.hparams.get("num_dfas"),
            dfa_emb_dim=self.hparams.get("dfa_emb_dim", 16),
            padding_idx=self.hparams["padding_idx"],
        )

        # Cargar pesos
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def encode(
        self,
        prefix_ids: LongTensor,
        lengths: LongTensor,
        dfa_ids: LongTensor | None = None,
    ) -> Tensor:
        """
        Codifica prefijos en estados.

        Args:
            prefix_ids: [batch, seq_len] IDs de tokens
            lengths: [batch] Longitudes reales
            dfa_ids: [batch] IDs de DFA (opcional, solo si use_dfa_embedding=True)

        Returns:
            h_t: [batch, d_state] Estados codificados
        """
        prefix_ids = prefix_ids.to(self.device)
        lengths = lengths.to(self.device)
        if dfa_ids is not None:
            dfa_ids = dfa_ids.to(self.device)

        with torch.no_grad():
            h_t = self.model(prefix_ids, lengths, dfa_ids)

        return h_t

    def encode_batch(
        self,
        prefix_ids_list: list[list[int]],
        lengths_list: list[int] | None = None,
        dfa_ids_list: list[int] | None = None,
    ) -> Tensor:
        """
        Codifica un batch de prefijos desde listas de Python.

        Args:
            prefix_ids_list: Lista de listas de IDs de tokens
            lengths_list: Lista de longitudes (si None, se calculan automáticamente)
            dfa_ids_list: Lista de IDs de DFA (opcional)

        Returns:
            h_t: [batch, d_state] Estados codificados
        """
        batch_size = len(prefix_ids_list)
        max_len = self.hparams["max_len"]
        padding_idx = self.hparams["padding_idx"]

        # Convertir a tensores
        prefix_ids_tensor = torch.full((batch_size, max_len), padding_idx, dtype=torch.long)
        if lengths_list is None:
            lengths_tensor = torch.zeros(batch_size, dtype=torch.long)
        else:
            lengths_tensor = torch.tensor(lengths_list, dtype=torch.long)

        for i, prefix_ids in enumerate(prefix_ids_list):
            seq_len = min(len(prefix_ids), max_len)
            prefix_ids_tensor[i, :seq_len] = torch.tensor(prefix_ids[:seq_len], dtype=torch.long)
            if lengths_list is None:
                lengths_tensor[i] = seq_len

        dfa_ids_tensor = None
        if dfa_ids_list is not None:
            dfa_ids_tensor = torch.tensor(dfa_ids_list, dtype=torch.long)

        return self.encode(prefix_ids_tensor, lengths_tensor, dfa_ids_tensor)


def load_state_encoder(
    model_path: Path | str,
    hparams_path: Path | str,
    device: str | None = None,
) -> StateEncoderInference:
    """
    Función de conveniencia para cargar StateEncoder.

    Args:
        model_path: Ruta a state_encoder.pth
        hparams_path: Ruta a statenet_hparams.json
        device: Dispositivo (cuda/cpu)

    Returns:
        StateEncoderInference listo para usar
    """
    return StateEncoderInference(
        Path(model_path), Path(hparams_path), device
    )

