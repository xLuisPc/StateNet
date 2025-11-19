#!/usr/bin/env python3
"""
StateNet - Generación de Estados Discretos (Colab y Local)
Combina generación de embeddings, discretización, validación y exportación.
Funciona tanto en Google Colab como en local usando rutas relativas.
"""

import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None

# ============================================================================
# DETECCIÓN DE ENTORNO Y RUTAS
# ============================================================================

def get_base_path():
    """Detecta si estamos en Colab o local y retorna la ruta base."""
    # Detectar si estamos en Colab
    if os.path.exists("/content"):
        # En Colab: usar /content/StateNet
        base = Path("/content/StateNet")
    else:
        # En local: usar ruta relativa desde este script
        # Este script está en scripts/, subimos 1 nivel para llegar a la raíz
        base = Path(__file__).resolve().parents[1]
    return base

BASE_PATH = get_base_path()

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

CONFIG = {
    "k_max": 16,
    "tau_accept": 0.5,
    "batch_size_embeddings": 256,
    "merge_threshold_cosine": 0.95,
    "merge_threshold_delta": 0.9,
    "coverage_samples": 100,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# ============================================================================
# MODELO STATEENCODER (simplificado para inferencia)
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


def load_model(model_path, hparams_path, device):
    """Carga modelo entrenado."""
    with open(hparams_path, 'r') as f:
        hparams = json.load(f)
    
    model = StateEncoder(
        vocab_size=hparams["vocab_size"],
        emb_dim=hparams["emb_dim"],
        hidden_dim=hparams["hidden_dim"],
        d_state=hparams["d_state"],
        max_len=hparams["max_len"],
        num_layers=hparams["num_layers"],
        rnn_type=hparams["rnn_type"],
        use_dfa_embedding=hparams.get("use_dfa_embedding", False),
        num_dfas=hparams.get("num_dfas"),
        dfa_emb_dim=hparams.get("dfa_emb_dim", 16),
        padding_idx=hparams["padding_idx"],
    )
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, hparams


# ============================================================================
# GENERACIÓN DE EMBEDDINGS
# ============================================================================

def generate_embeddings(model, df, vocab, max_len, batch_size, device):
    """Genera embeddings h_t para todos los prefijos."""
    print(f"Generando embeddings para {len(df)} prefijos...")
    
    prefix_ids_list = []
    lengths_list = []
    dfa_ids_list = []
    
    for _, row in df.iterrows():
        prefix_ids = json.loads(row["prefix_ids"])
        length = int(row["length"])
        dfa_id = int(row["dfa_id"])
        
        if len(prefix_ids) > max_len:
            prefix_ids = prefix_ids[:max_len]
            length = max_len
        else:
            padding = [vocab["<PAD>"]] * (max_len - len(prefix_ids))
            prefix_ids = prefix_ids + padding
        
        prefix_ids_list.append(prefix_ids)
        lengths_list.append(length)
        dfa_ids_list.append(dfa_id)
    
    h_t_list = []
    n_batches = (len(df) + batch_size - 1) // batch_size
    
    use_dfa_emb = model.use_dfa_embedding
    
    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            batch_prefix_ids = prefix_ids_list[i:i+batch_size]
            batch_lengths = lengths_list[i:i+batch_size]
            batch_dfa_ids = dfa_ids_list[i:i+batch_size] if use_dfa_emb else None
            
            prefix_ids_tensor = torch.tensor(batch_prefix_ids, dtype=torch.long).to(device)
            lengths_tensor = torch.tensor(batch_lengths, dtype=torch.long).to(device)
            dfa_ids_tensor = torch.tensor(batch_dfa_ids, dtype=torch.long).to(device) if batch_dfa_ids else None
            
            h_t = model(prefix_ids_tensor, lengths_tensor, dfa_ids_tensor)
            h_t_list.append(h_t.cpu().numpy())
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"  Procesado {i+len(batch_prefix_ids)}/{len(df)} prefijos")
    
    h_t_all = np.vstack(h_t_list)
    df = df.copy()
    df["h_t"] = [h_t_all[i] for i in range(len(df))]
    
    print(f"  ✓ Embeddings generados: shape {h_t_all.shape}")
    return df


# ============================================================================
# DISCRETIZACIÓN CON K-MEANS
# ============================================================================

def discretize_states(df, k_max):
    """Discretiza estados por dfa_id usando k-means."""
    print(f"\nDiscretizando estados con K_max={k_max}...")
    
    results = {}
    unique_dfas = sorted(df["dfa_id"].unique())
    
    for dfa_id in unique_dfas:
        dfa_df = df[df["dfa_id"] == dfa_id].copy()
        h_t_array = np.array([h for h in dfa_df["h_t"]])
        n_samples = len(h_t_array)
        
        if n_samples == 0:
            continue
        
        d_state = h_t_array.shape[1]
        k_actual = min(k_max, n_samples)
        
        kmeans = KMeans(n_clusters=k_actual, random_state=42, n_init=10)
        assignments = kmeans.fit_predict(h_t_array)
        
        centroids = np.zeros((k_max, d_state))
        centroids[:k_actual] = kmeans.cluster_centers_
        
        m_use = np.zeros(k_max, dtype=bool)
        unique_clusters = np.unique(assignments)
        for q in unique_clusters:
            if 0 <= q < k_max:
                m_use[q] = True
        
        results[dfa_id] = {
            "centroids": centroids,
            "m_use": m_use,
            "state_assignments": assignments,
        }
        
        print(f"  DFA {dfa_id}: {n_samples} prefijos → {k_actual} clusters, {m_use.sum()} estados usados")
    
    print(f"  ✓ Discretización completada para {len(results)} DFAs")
    return results


# ============================================================================
# CONSTRUCCIÓN DE TRANSICIONES
# ============================================================================

def build_transitions(df, discretization, vocab):
    """Construye tabla de transiciones δ(q, a) = q'."""
    print("\nConstruyendo tabla de transiciones...")
    
    symbols = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "<EOS>"]
    symbol_to_idx = {sym: idx for idx, sym in enumerate(symbols)}
    num_symbols = len(symbols)
    
    transitions = {}
    
    for dfa_id, disc_data in discretization.items():
        dfa_df = df[df["dfa_id"] == dfa_id].copy().reset_index(drop=True)
        assignments = disc_data["state_assignments"]
        k_max = disc_data["centroids"].shape[0]
        
        prefix_to_state = {}
        for i, (_, row) in enumerate(dfa_df.iterrows()):
            prefix = str(row["prefix"])
            q = assignments[i]
            prefix_to_state[prefix] = q
        
        delta = np.full((k_max, num_symbols), -1, dtype=np.int32)
        transition_counts = defaultdict(lambda: defaultdict(int))
        
        for i, (_, row) in enumerate(dfa_df.iterrows()):
            q = assignments[i]
            next_symbol = str(row["next_symbol"])
            prefix = str(row["prefix"])
            
            if next_symbol == "<EOS>" or next_symbol not in symbol_to_idx:
                continue
            
            a_idx = symbol_to_idx[next_symbol]
            
            if prefix == "<EPS>":
                next_prefix = next_symbol
            else:
                next_prefix = prefix + next_symbol
            
            if next_prefix in prefix_to_state:
                q_prime = prefix_to_state[next_prefix]
                transition_counts[(q, a_idx)][q_prime] += 1
        
        for (q, a_idx), q_prime_counts in transition_counts.items():
            if q_prime_counts:
                q_prime = max(q_prime_counts.items(), key=lambda x: x[1])[0]
                delta[q, a_idx] = q_prime
        
        transitions[dfa_id] = delta
        
        n_observed = (delta >= 0).sum()
        n_total = k_max * num_symbols
        print(f"  DFA {dfa_id}: {n_observed}/{n_total} transiciones observadas ({100*n_observed/n_total:.1f}%)")
    
    print(f"  ✓ Tabla de transiciones construida para {len(transitions)} DFAs")
    return transitions


# ============================================================================
# MARCAR ESTADOS DE ACEPTACIÓN
# ============================================================================

def mark_accept_states(df, discretization, tau_accept):
    """Marca estados de aceptación basado en ratio de cadenas aceptadas."""
    print(f"\nMarcando estados de aceptación (τ={tau_accept})...")
    
    accept_states = {}
    
    for dfa_id, disc_data in discretization.items():
        dfa_df = df[df["dfa_id"] == dfa_id].copy().reset_index(drop=True)
        assignments = disc_data["state_assignments"]
        k_max = disc_data["centroids"].shape[0]
        
        state_counts = defaultdict(lambda: {"total": 0, "accepted": 0})
        
        prefix_to_state = {}
        for i, (_, row) in enumerate(dfa_df.iterrows()):
            prefix = str(row["prefix"])
            q = assignments[i]
            prefix_to_state[prefix] = q
        
        string_groups = dfa_df.groupby("string")
        for string, group in string_groups:
            final_row = group.iloc[-1]
            final_label = float(final_row["final_label"])
            final_prefix = str(final_row["prefix"])
            
            if final_prefix in prefix_to_state:
                q = prefix_to_state[final_prefix]
                state_counts[q]["total"] += 1
                if final_label > 0.5:
                    state_counts[q]["accepted"] += 1
        
        m_accept = np.zeros(k_max, dtype=bool)
        for q in range(k_max):
            if q in state_counts:
                ratio = state_counts[q]["accepted"] / state_counts[q]["total"]
                m_accept[q] = ratio >= tau_accept
        
        accept_states[dfa_id] = m_accept
        n_accept = m_accept.sum()
        print(f"  DFA {dfa_id}: {n_accept}/{k_max} estados de aceptación")
    
    print(f"  ✓ Estados de aceptación marcados para {len(accept_states)} DFAs")
    return accept_states


# ============================================================================
# MERGE DE ESTADOS
# ============================================================================

def merge_states(states_per_dfa, threshold_cosine=0.95, threshold_delta=0.9):
    """Fusiona estados similares."""
    print(f"\nFusionando estados similares...")
    print(f"  Umbral coseno: {threshold_cosine}")
    print(f"  Umbral delta: {threshold_delta}")
    
    merged_states = {}
    
    for dfa_id, data in states_per_dfa.items():
        centroids = data["centroids"]
        m_use = data["m_use"]
        m_accept = data["m_accept"]
        delta = data["delta"]
        
        k_max = centroids.shape[0]
        used_states = np.where(m_use)[0]
        
        if len(used_states) == 0:
            merged_states[dfa_id] = data
            continue
        
        state_map = {q: q for q in used_states}
        to_merge = []
        
        for i, q1 in enumerate(used_states):
            if q1 not in state_map:
                continue
            
            for q2 in used_states[i+1:]:
                if q2 not in state_map:
                    continue
                
                if m_accept[q1] != m_accept[q2]:
                    continue
                
                delta1 = delta[q1, :]
                delta2 = delta[q2, :]
                valid_mask = (delta1 >= 0) & (delta2 >= 0)
                if valid_mask.sum() == 0:
                    continue
                delta_sim = (delta1[valid_mask] == delta2[valid_mask]).sum() / valid_mask.sum()
                if delta_sim < threshold_delta:
                    continue
                
                c1 = centroids[q1]
                c2 = centroids[q2]
                norm1 = np.linalg.norm(c1)
                norm2 = np.linalg.norm(c2)
                if norm1 == 0 or norm2 == 0:
                    continue
                cosine_sim = np.dot(c1, c2) / (norm1 * norm2)
                if cosine_sim < threshold_cosine:
                    continue
                
                to_merge.append((q1, q2))
                state_map[q2] = q1
        
        merge_groups = defaultdict(list)
        for q in used_states:
            target = state_map[q]
            merge_groups[target].append(q)
        
        new_centroids = centroids.copy()
        new_delta = delta.copy()
        new_m_use = m_use.copy()
        new_m_accept = m_accept.copy()
        
        for target, group in merge_groups.items():
            if len(group) > 1:
                new_centroids[target] = centroids[group].mean(axis=0)
                
                for a in range(delta.shape[1]):
                    values = [delta[q, a] for q in group if delta[q, a] >= 0]
                    if values:
                        most_common = Counter(values).most_common(1)[0][0]
                        new_delta[target, a] = most_common
                
                for q in group:
                    if q != target:
                        new_m_use[q] = False
        
        final_used = np.where(new_m_use)[0]
        if len(final_used) == 0:
            merged_states[dfa_id] = {
                "centroids": new_centroids,
                "m_use": new_m_use,
                "m_accept": new_m_accept,
                "delta": new_delta,
            }
            continue
        
        compact_map = {old_q: new_q for new_q, old_q in enumerate(final_used)}
        k_compact = len(final_used)
        
        compact_centroids = np.zeros((k_max, centroids.shape[1]))
        compact_m_use = np.zeros(k_max, dtype=bool)
        compact_m_accept = np.zeros(k_max, dtype=bool)
        compact_delta = np.full((k_max, delta.shape[1]), -1, dtype=np.int32)
        
        for old_q, new_q in compact_map.items():
            compact_centroids[new_q] = new_centroids[old_q]
            compact_m_use[new_q] = True
            compact_m_accept[new_q] = new_m_accept[old_q]
            
            for a in range(delta.shape[1]):
                old_target = new_delta[old_q, a]
                if old_target >= 0 and old_target in compact_map:
                    compact_delta[new_q, a] = compact_map[old_target]
        
        merged_states[dfa_id] = {
            "centroids": compact_centroids,
            "m_use": compact_m_use,
            "m_accept": compact_m_accept,
            "delta": compact_delta,
        }
        
        n_merged = len(to_merge)
        n_final = compact_m_use.sum()
        print(f"  DFA {dfa_id}: {len(used_states)} → {n_final} estados ({n_merged} fusiones)")
    
    print(f"  ✓ Merge completado para {len(merged_states)} DFAs")
    return merged_states


# ============================================================================
# VALIDACIÓN CON ALPHABETNET (OPCIONAL)
# ============================================================================

def validate_with_alphabetnet(states_per_dfa, alphabet_pred_path, vocab):
    """Valida y corrige transiciones usando AlphabetNet."""
    print("\nValidando con AlphabetNet...")
    
    if not os.path.exists(alphabet_pred_path):
        print(f"  ⚠ Archivo no encontrado: {alphabet_pred_path}")
        print(f"  Saltando validación con AlphabetNet")
        return states_per_dfa
    
    with open(alphabet_pred_path, 'r') as f:
        alphabet_pred = json.load(f)
    
    symbols = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "<EOS>"]
    idx_to_symbol = {idx: sym for idx, sym in enumerate(symbols)}
    
    validated_states = {}
    total_corrections = 0
    
    for dfa_id, data in states_per_dfa.items():
        delta = data["delta"].copy()
        m_use = data["m_use"]
        
        dfa_key = str(dfa_id)
        if dfa_key not in alphabet_pred:
            validated_states[dfa_id] = data
            continue
        
        predicted_alphabet = set(alphabet_pred[dfa_key])
        n_corrections = 0
        
        used_states = np.where(m_use)[0]
        for q in used_states:
            for j in range(delta.shape[1]):
                if delta[q, j] == -1:
                    continue
                
                if j >= len(symbols):
                    continue
                symbol = idx_to_symbol[j]
                
                if symbol not in predicted_alphabet:
                    delta[q, j] = -1
                    n_corrections += 1
        
        validated_states[dfa_id] = {**data, "delta": delta}
        total_corrections += n_corrections
        if n_corrections > 0:
            print(f"  DFA {dfa_id}: {n_corrections} transiciones corregidas")
    
    print(f"  ✓ Validación completada: {total_corrections} correcciones totales")
    return validated_states


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def run_pipeline(
    model_path,
    hparams_path,
    vocab_path,
    train_csv_path,
    val_csv_path,
    output_path="states_for_acceptnet.pt",
    test_csv_path=None,
    alphabet_pred_path=None,
    config=None
):
    """
    Ejecuta el pipeline completo.
    
    Args:
        model_path: Ruta a state_encoder.pth
        hparams_path: Ruta a statenet_hparams.json
        vocab_path: Ruta a vocab_char_to_id.json
        train_csv_path: Ruta a prefix_train.csv
        val_csv_path: Ruta a prefix_val.csv
        output_path: Ruta de salida (default: states_for_acceptnet.pt)
        test_csv_path: Opcional, ruta a prefix_test.csv
        alphabet_pred_path: Opcional, ruta a alphabet_pred.json
        config: Dict con configuración (usa CONFIG por defecto)
    """
    if config is None:
        config = CONFIG
    
    device = torch.device(config["device"])
    print(f"🚀 Iniciando pipeline en {device}")
    print(f"Configuración: {config}\n")

    start_time = time.time()
    process = psutil.Process(os.getpid()) if psutil else None
    mem_start = process.memory_info().rss if process else None
    if psutil:
        psutil.cpu_percent(interval=None)  # inicializa medición

    gpu_enabled = device.type == "cuda" and torch.cuda.is_available()
    if gpu_enabled:
        torch.cuda.reset_peak_memory_stats()
        print(f"💻 GPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   Memoria total GPU: {total_mem:.2f} GB\n")
    
    # 1. Cargar vocabulario
    print("1. Cargando vocabulario...")
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    print(f"  ✓ Vocab size: {len(vocab)}\n")
    
    # 2. Cargar modelo
    print("2. Cargando modelo...")
    model, hparams = load_model(model_path, hparams_path, device)
    max_len = hparams["max_len"]
    print(f"  ✓ Modelo cargado (d_state={hparams['d_state']}, max_len={max_len})\n")
    
    # 3. Cargar datasets
    print("3. Cargando datasets...")
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    print(f"  Train: {len(train_df)} prefijos")
    print(f"  Val: {len(val_df)} prefijos")
    
    if test_csv_path and os.path.exists(test_csv_path):
        test_df = pd.read_csv(test_csv_path)
        print(f"  Test: {len(test_df)} prefijos")
        df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    else:
        df = pd.concat([train_df, val_df], ignore_index=True)
    
    print(f"  Total: {len(df)} prefijos\n")
    
    # 4. Generar embeddings
    print("4. Generando embeddings...")
    df_with_embeddings = generate_embeddings(
        model, df, vocab, max_len, config["batch_size_embeddings"], device
    )
    print()
    
    # 5. Discretizar estados
    print("5. Discretizando estados...")
    discretization = discretize_states(df_with_embeddings, config["k_max"])
    print()
    
    # 6. Construir transiciones
    print("6. Construyendo transiciones...")
    transitions = build_transitions(df_with_embeddings, discretization, vocab)
    print()
    
    # 7. Marcar estados de aceptación
    print("7. Marcando estados de aceptación...")
    accept_states = mark_accept_states(df_with_embeddings, discretization, config["tau_accept"])
    print()
    
    # 8. Construir estructura inicial
    print("8. Construyendo estructura inicial...")
    states_per_dfa = {}
    for dfa_id in discretization.keys():
        states_per_dfa[dfa_id] = {
            "centroids": discretization[dfa_id]["centroids"],
            "m_use": discretization[dfa_id]["m_use"],
            "m_accept": accept_states[dfa_id],
            "delta": transitions[dfa_id],
        }
    print()
    
    # 9. Merge de estados
    print("9. Fusionando estados similares...")
    merged_states = merge_states(
        states_per_dfa,
        threshold_cosine=config["merge_threshold_cosine"],
        threshold_delta=config["merge_threshold_delta"],
    )
    print()
    
    # 10. Validación con AlphabetNet (opcional)
    if alphabet_pred_path:
        print("10. Validando con AlphabetNet...")
        validated_states = validate_with_alphabetnet(merged_states, alphabet_pred_path, vocab)
        print()
    else:
        validated_states = merged_states
    
    # 11. Exportar formato final
    print("11. Exportando formato final...")
    
    # Crear directorio de salida si no existe
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    states_for_acceptnet = {}
    for dfa_id, data in validated_states.items():
        # Convertir arrays de numpy a tensores de PyTorch para compatibilidad
        centroids = data["centroids"]
        m_use = data["m_use"]
        m_accept = data["m_accept"]
        delta = data["delta"]
        
        # Convertir a tensores si son arrays de numpy
        if isinstance(centroids, np.ndarray):
            centroids = torch.from_numpy(centroids).float()
        elif isinstance(centroids, torch.Tensor):
            centroids = centroids.cpu().float()
        
        if isinstance(m_use, np.ndarray):
            m_use = torch.from_numpy(m_use).bool()
        elif isinstance(m_use, torch.Tensor):
            m_use = m_use.cpu().bool()
        
        if isinstance(m_accept, np.ndarray):
            m_accept = torch.from_numpy(m_accept).bool()
        elif isinstance(m_accept, torch.Tensor):
            m_accept = m_accept.cpu().bool()
        
        if isinstance(delta, np.ndarray):
            delta = torch.from_numpy(delta).long()
        elif isinstance(delta, torch.Tensor):
            delta = delta.cpu().long()
        
        states_for_acceptnet[dfa_id] = {
            "E": centroids,
            "m_use": m_use,
            "m_accept": m_accept,
            "delta": delta,
        }
    
    torch.save(states_for_acceptnet, output_path)
    print(f"  ✓ Guardado en {output_path}")
    print(f"  {len(states_for_acceptnet)} DFAs exportados\n")
    
    elapsed = time.time() - start_time
    print("✅ Pipeline completado exitosamente!")
    print(f"⏱️  Tiempo total: {elapsed / 60:.2f} min ({elapsed:.1f} s)")
    
    if process:
        mem_end = process.memory_info().rss
        delta_gb = (mem_end - mem_start) / 1e9 if mem_start is not None else 0.0
        print(f"🧠 RAM proceso: {mem_end / 1e9:.2f} GB (Δ {delta_gb:+.2f} GB)")
    if psutil:
        cpu_usage = psutil.cpu_percent(interval=0.3)
        vm = psutil.virtual_memory()
        print(f"⚙️  CPU promedio (último muestreo): {cpu_usage:.1f}%")
        print(f"📦 RAM sistema: usada {vm.used / 1e9:.2f} GB / total {vm.total / 1e9:.2f} GB")
    
    if gpu_enabled:
        current_alloc = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        peak_alloc = torch.cuda.max_memory_allocated(0) / 1e9
        print(f"🎮 GPU RAM actual: {current_alloc:.2f} GB (reservada {reserved:.2f} GB)")
        print(f"   Pico de uso GPU: {peak_alloc:.2f} GB")
    
    return states_for_acceptnet


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    # Ejemplo de uso (funciona en Colab y local automáticamente)
    run_pipeline(
        model_path=str(BASE_PATH / "checkpoints" / "state_encoder.pth"),
        hparams_path=str(BASE_PATH / "checkpoints" / "statenet_hparams.json"),
        vocab_path=str(BASE_PATH / "vocab_char_to_id.json"),
        train_csv_path=str(BASE_PATH / "data" / "statenet" / "prefix_train.csv"),
        val_csv_path=str(BASE_PATH / "data" / "statenet" / "prefix_val.csv"),
        output_path=str(BASE_PATH / "artifacts" / "statenet" / "states_for_acceptnet.pt"),
        test_csv_path=str(BASE_PATH / "data" / "statenet" / "prefix_test.csv"),  # Opcional
        alphabet_pred_path=None,  # Opcional: str(BASE_PATH / "artifacts" / "alphabetnet" / "alphabet_pred.json")
        config=CONFIG
    )

