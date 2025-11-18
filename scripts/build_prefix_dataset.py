#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd


ALLOWED_SYMBOLS = set("ABCDEFGHIJKL")
MAX_LEN_DEFAULT = 64
RNG_SEED = 1337


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generar dataset de prefijos para StateNet.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=root / "data" / "dataset6000.csv",
        help="Ruta al CSV original (con columnas Regex, Alfabeto, Clase, ...).",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=root / "vocab_char_to_id.json",
        help="Ruta al vocabulario compartido con AlphabetNet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data" / "statenet",
        help="Directorio donde se guardarán los archivos resultantes.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=MAX_LEN_DEFAULT,
        help="Longitud máxima (tokens) del prefijo codificado.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RNG_SEED,
        help="Semilla para el split estratificado por dfa_id.",
    )
    return parser.parse_args()


def load_vocab(vocab_path: Path) -> Dict[str, int]:
    with vocab_path.open("r", encoding="utf-8") as f:
        vocab = json.load(f)
    required = {"<PAD>", "<EPS>"}
    if missing := sorted(required - set(vocab.keys())):
        raise ValueError(f"El vocabulario no contiene los símbolos requeridos: {missing}")
    return vocab


def normalize_string(raw: str) -> str:
    if raw in {"", "<EPS>"}:
        return "<EPS>"
    if any(ch not in ALLOWED_SYMBOLS for ch in raw):
        invalid = sorted({ch for ch in raw if ch not in ALLOWED_SYMBOLS})
        raise ValueError(f"Cadena con caracteres inválidos {invalid}: {raw}")
    return raw


def load_flat_dataset(dataset_path: Path) -> pd.DataFrame:
    flat_rows = []
    with dataset_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for dfa_id, row in enumerate(reader):
            cls_field = row.get("Clase")
            if not cls_field:
                continue
            label_map = json.loads(cls_field)
            for string, label in label_map.items():
                normalized = normalize_string(string)
                label_int = int(bool(label))
                if label_int not in (0, 1):
                    raise ValueError(f"Etiqueta inválida {label} en dfa_id={dfa_id} string={string}")
                flat_rows.append(
                    {
                        "dfa_id": dfa_id,
                        "string": normalized,
                        "label": label_int,
                    }
                )
    if not flat_rows:
        raise RuntimeError("No se generaron filas al procesar el dataset base.")
    return pd.DataFrame(flat_rows)


def encode_prefix(prefix: str, vocab: Dict[str, int], max_len: int) -> tuple[List[int], int]:
    tokens: Sequence[str]
    if prefix == "<EPS>":
        tokens = ["<EPS>"]
    else:
        tokens = list(prefix)
    ids = [vocab[token] for token in tokens]
    if len(ids) > max_len:
        ids = ids[:max_len]
    length = len(ids)
    pad_id = vocab["<PAD>"]
    padded = ids + [pad_id] * (max_len - len(ids))
    return padded, length


def build_prefix_rows(flat_df: pd.DataFrame, vocab: Dict[str, int], max_len: int) -> pd.DataFrame:
    rows = []
    for item in flat_df.itertuples(index=False):
        if item.string == "<EPS>":
            prefixes = ["<EPS>"]
            next_symbols = ["<EOS>"]
        else:
            prefixes = ["<EPS>"] + [item.string[:i] for i in range(1, len(item.string) + 1)]
            next_symbols = list(item.string) + ["<EOS>"]
        for prefix, next_symbol in zip(prefixes, next_symbols):
            encoded, length = encode_prefix(prefix, vocab, max_len)
            rows.append(
                {
                    "dfa_id": item.dfa_id,
                    "string": item.string,
                    "prefix": prefix,
                    "next_symbol": next_symbol,
                    "final_label": item.label,
                    "prefix_ids": encoded,
                    "length": length,
                }
            )
    return pd.DataFrame(rows)


def split_dfa_ids(dfa_ids: Iterable[int], seed: int) -> Dict[str, List[int]]:
    unique_ids = sorted({int(x) for x in dfa_ids})
    rng = random.Random(seed)
    rng.shuffle(unique_ids)
    n = len(unique_ids)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train_ids = unique_ids[:n_train]
    val_ids = unique_ids[n_train : n_train + n_val]
    test_ids = unique_ids[n_train + n_val :]
    return {
        "train": sorted(int(x) for x in train_ids),
        "val": sorted(int(x) for x in val_ids),
        "test": sorted(int(x) for x in test_ids),
    }


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def compute_vocab_hash(vocab_path: Path) -> str:
    data = vocab_path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    args = parse_args()
    vocab = load_vocab(args.vocab)
    flat_df = load_flat_dataset(args.dataset)
    prefix_df = build_prefix_rows(flat_df, vocab, args.max_len)
    prefix_df["prefix_ids"] = prefix_df["prefix_ids"].apply(json.dumps)
    split_map = split_dfa_ids(prefix_df["dfa_id"].unique(), args.seed)
    split_lookup = {
        dfa_id: split
        for split, ids in split_map.items()
        for dfa_id in ids
    }
    prefix_df["split"] = prefix_df["dfa_id"].map(split_lookup)
    if prefix_df["split"].isna().any():
        raise RuntimeError("Existen filas sin split asignado.")

    output_dir = args.output_dir
    save_csv(prefix_df[prefix_df["split"] == "train"], output_dir / "prefix_train.csv")
    save_csv(prefix_df[prefix_df["split"] == "val"], output_dir / "prefix_val.csv")
    save_csv(prefix_df[prefix_df["split"] == "test"], output_dir / "prefix_test.csv")

    meta = {
        "dataset_source": str(args.dataset.resolve()),
        "max_len": args.max_len,
        "vocab_file": str(args.vocab.name),
        "vocab_sha256": compute_vocab_hash(args.vocab),
        "output_format": "csv",
        "dataset_files": {
            "train": "prefix_train.csv",
            "val": "prefix_val.csv",
            "test": "prefix_test.csv",
        },
        "splits": split_map,
        "num_prefix_samples": {
            "train": int((prefix_df["split"] == "train").sum()),
            "val": int((prefix_df["split"] == "val").sum()),
            "test": int((prefix_df["split"] == "test").sum()),
        },
    }
    meta_path = output_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Guardados datasets en {output_dir}")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

