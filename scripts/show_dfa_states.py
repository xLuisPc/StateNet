l #!/usr/bin/env python3
"""
Consulta los estados discretos generados para un DFA específico.

Permite buscar por regex (tal como aparece en dataset6000.csv) o directamente
por dfa_id, y muestra la información almacenada en states_for_acceptnet.pt.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATES = ROOT / "artifacts" / "statenet" / "states_for_acceptnet.pt"
DEFAULT_DATASET = ROOT / "data" / "dataset6000.csv"
SYMBOLS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "<EOS>"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mostrar estados para un DFA/regex.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--regex", type=str, help="Regex exactamente como aparece en dataset6000.csv")
    group.add_argument("--dfa-id", type=int, help="ID numérico del DFA")

    parser.add_argument(
        "--states-path",
        type=Path,
        default=DEFAULT_STATES,
        help=f"Ruta al archivo states_for_acceptnet.pt (default: {DEFAULT_STATES})",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Ruta a dataset6000.csv para mapear regex→dfa_id (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--show-delta",
        action="store_true",
        help="Imprimir la tabla completa de transiciones (puede ser extensa).",
    )
    parser.add_argument(
        "--max-states",
        type=int,
        default=5,
        help="Número de estados a mostrar cuando no se usa --show-delta (default: 5).",
    )
    return parser.parse_args()


def load_regex_mapping(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"No se encuentra el dataset base: {dataset_path}")
    # Mantener orden original para que dfa_id coincida con enumerate
    df = pd.read_csv(dataset_path, usecols=["Regex", "Alfabeto"])
    df = df.reset_index().rename(columns={"index": "dfa_id"})
    return df


def resolve_dfa_id(args: argparse.Namespace, mapping_df: pd.DataFrame) -> tuple[int, str, str]:
    if args.dfa_id is not None:
        if args.dfa_id < 0 or args.dfa_id >= len(mapping_df):
            raise ValueError(f"dfa_id fuera de rango (0-{len(mapping_df)-1}): {args.dfa_id}")
        row = mapping_df.iloc[args.dfa_id]
        return int(row["dfa_id"]), row["Regex"], row["Alfabeto"]

    matches = mapping_df[mapping_df["Regex"] == args.regex]
    if matches.empty:
        raise ValueError(f"No se encontró la regex exactamente igual en {args.dataset_path}: {args.regex!r}")
    if len(matches) > 1:
        ids = matches["dfa_id"].tolist()
        raise ValueError(
            f"La regex aparece múltiples veces ({len(matches)}). "
            f"Especifica --dfa-id para desambiguar. IDs encontrados: {ids}"
        )
    row = matches.iloc[0]
    return int(row["dfa_id"]), row["Regex"], row["Alfabeto"]


def describe_transitions(delta_row: torch.Tensor) -> str:
    parts = []
    for idx, target in enumerate(delta_row.tolist()):
        if target >= 0:
            parts.append(f"{SYMBOLS[idx]}→{target}")
    return ", ".join(parts) if parts else "Sin transiciones observadas"


def main() -> None:
    args = parse_args()

    mapping_df = load_regex_mapping(args.dataset_path)
    dfa_id, regex, alphabet = resolve_dfa_id(args, mapping_df)

    if not args.states_path.exists():
        raise FileNotFoundError(f"No se encuentra el archivo de estados: {args.states_path}")

    states_dict = torch.load(args.states_path)
    if dfa_id not in states_dict:
        # Por compatibilidad, intentar con string como clave
        dfa_key = str(dfa_id)
        if dfa_key not in states_dict:
            available = sorted(states_dict.keys())
            raise KeyError(f"DFA {dfa_id} no está en {args.states_path}. Claves disponibles: {available[:5]}...")
        dfa_data = states_dict[dfa_key]
    else:
        dfa_data = states_dict[dfa_id]

    E = dfa_data["E"]
    m_use = dfa_data["m_use"]
    m_accept = dfa_data["m_accept"]
    delta = dfa_data["delta"]

    used_indices = m_use.nonzero(as_tuple=True)[0].tolist()
    accept_indices = [idx for idx in used_indices if bool(m_accept[idx].item())]

    print("=" * 60)
    print("📄  Información del DFA")
    print("=" * 60)
    print(f"DFA ID           : {dfa_id}")
    print(f"Regex            : {regex}")
    print(f"Alfabeto         : {alphabet}")
    print(f"Estados usados   : {len(used_indices)} / {len(m_use)}")
    print(f"Estados aceptación: {accept_indices if accept_indices else 'Ninguno'}")
    print(f"E (centroides) shape: {tuple(E.shape)}")
    print()

    if args.show_delta:
        print("δ(q, símbolo) -> q'")
        print("-" * 60)
        for idx in used_indices:
            transitions = describe_transitions(delta[idx])
            print(f"q{idx}: {transitions}")
    else:
        print(f"Transiciones (primeros {args.max_states} estados usados):")
        print("-" * 60)
        for idx in used_indices[: args.max_states]:
            transitions = describe_transitions(delta[idx])
            print(f"q{idx}: {transitions}")
        remaining = len(used_indices) - args.max_states
        if remaining > 0:
            print(f"... {remaining} estados adicionales (usa --show-delta para ver todos)")


if __name__ == "__main__":
    main()

