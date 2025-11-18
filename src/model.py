#!/usr/bin/env python3
"""Arquitectura StateEncoder para codificar prefijos en estados de DFA."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import LongTensor, Tensor


class StateEncoder(nn.Module):
    """
    Codifica prefijos de cadenas en estados de DFA.

    Entradas:
        prefix_ids: LongTensor[batch, seq_len] - IDs de tokens del prefijo
        lengths: LongTensor[batch] - Longitudes reales de cada secuencia
        dfa_ids: LongTensor[batch] (opcional) - IDs de DFA para embedding

    Salida:
        h_t: Tensor[batch, d_state] - Estado codificado final
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        d_state: int,
        max_len: int = 64,
        num_layers: int = 1,
        rnn_type: str = "GRU",
        use_dfa_embedding: bool = False,
        num_dfas: int | None = None,
        dfa_emb_dim: int = 16,
        padding_idx: int = 0,
    ):
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

        # Embedding de tokens
        self.token_embedding = nn.Embedding(
            vocab_size, emb_dim, padding_idx=padding_idx
        )

        # RNN
        if self.rnn_type == "GRU":
            self.rnn = nn.GRU(
                emb_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=False,
            )
        elif self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                emb_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=False,
            )
        else:
            raise ValueError(f"rnn_type debe ser 'GRU' o 'LSTM', recibido: {rnn_type}")

        # Embedding de DFA (opcional)
        if use_dfa_embedding:
            if num_dfas is None:
                raise ValueError("num_dfas debe especificarse si use_dfa_embedding=True")
            self.dfa_embedding = nn.Embedding(num_dfas, dfa_emb_dim)
            rnn_output_dim = hidden_dim
        else:
            self.dfa_embedding = None
            rnn_output_dim = hidden_dim

        # Proyección final al espacio de estados
        self.state_proj = nn.Linear(rnn_output_dim + (dfa_emb_dim if use_dfa_embedding else 0), d_state)

    def forward(
        self,
        prefix_ids: LongTensor,
        lengths: LongTensor,
        dfa_ids: LongTensor | None = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            prefix_ids: [batch, seq_len] IDs de tokens
            lengths: [batch] Longitudes reales (sin padding)
            dfa_ids: [batch] IDs de DFA (opcional, solo si use_dfa_embedding=True)

        Returns:
            h_t: [batch, d_state] Estado codificado final
        """
        batch_size = prefix_ids.size(0)

        # Embedding de tokens
        # [batch, seq_len] -> [batch, seq_len, emb_dim]
        embedded = self.token_embedding(prefix_ids)

        # Pack para RNN (ignora padding)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # RNN
        if self.rnn_type == "GRU":
            packed_output, hidden = self.rnn(packed)
        else:  # LSTM
            packed_output, (hidden, _) = self.rnn(packed)

        # Obtener último estado válido
        # hidden: [num_layers, batch, hidden_dim] -> [batch, hidden_dim]
        if self.num_layers == 1:
            h_rnn = hidden.squeeze(0)  # [batch, hidden_dim]
        else:
            h_rnn = hidden[-1]  # Tomar última capa

        # Concatenar embedding de DFA si se usa
        if self.use_dfa_embedding:
            if dfa_ids is None:
                raise ValueError("dfa_ids requerido cuando use_dfa_embedding=True")
            dfa_emb = self.dfa_embedding(dfa_ids)  # [batch, dfa_emb_dim]
            h_concat = torch.cat([h_rnn, dfa_emb], dim=1)  # [batch, hidden_dim + dfa_emb_dim]
        else:
            h_concat = h_rnn  # [batch, hidden_dim]

        # Proyección final
        h_t = self.state_proj(h_concat)  # [batch, d_state]

        return h_t


class StateEncoderWithHeads(nn.Module):
    """
    StateEncoder con cabezas de predicción para entrenamiento.

    Predice:
        - next_symbol: siguiente símbolo (clasificación)
        - final_label: etiqueta final binaria (regresión/BCE)
    """

    def __init__(
        self,
        encoder: StateEncoder,
        vocab_size: int,
        num_symbols: int = 13,  # A-L + <EOS> = 13
    ):
        super().__init__()
        self.encoder = encoder
        self.vocab_size = vocab_size
        self.num_symbols = num_symbols

        # Head para next_symbol (clasificación)
        # Vocab: <PAD>=0, <EPS>=1, A=2, B=3, ..., L=13, <EOS>=14 (si existe)
        # Pero necesitamos mapear a símbolos válidos: A-L + <EOS>
        self.next_symbol_head = nn.Linear(encoder.d_state, num_symbols)

        # Head para final_label (binario)
        self.final_label_head = nn.Linear(encoder.d_state, 1)

    def forward(
        self,
        prefix_ids: LongTensor,
        lengths: LongTensor,
        dfa_ids: LongTensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass con predicciones.

        Returns:
            next_symbol_logits: [batch, num_symbols] Logits para next_symbol
            final_label_logits: [batch, 1] Logits para final_label (sigmoid)
        """
        h_t = self.encoder(prefix_ids, lengths, dfa_ids)  # [batch, d_state]

        next_symbol_logits = self.next_symbol_head(h_t)  # [batch, num_symbols]
        final_label_logits = self.final_label_head(h_t)  # [batch, 1]

        return next_symbol_logits, final_label_logits

