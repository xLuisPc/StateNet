#!/usr/bin/env python3
"""
Script simple para probar el modelo StateEncoder.
Muestra entrada y salida de forma clara.
"""

import json
import sys
import torch
from pathlib import Path

# Agregar raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importar el modelo (versión local o colab)
try:
    from src.infer import load_state_encoder
    USE_LOCAL = True
except ImportError:
    # Si no está disponible, usar versión de statenet_colab
    from scripts.colab.statenet_colab import load_model
    USE_LOCAL = False


def string_to_ids(string, vocab, max_len=64):
    """Convierte una cadena a IDs usando el vocabulario."""
    ids = []
    for char in string:
        if char in vocab:
            ids.append(vocab[char])
        else:
            print(f"⚠️  Caracter '{char}' no está en el vocabulario, usando <PAD>")
            ids.append(vocab.get("<PAD>", 0))
    
    # Truncar si es muy largo
    ids = ids[:max_len]
    return ids, len(ids)


def test_model_simple(prefix_string="ABC"):
    """
    Prueba el modelo con un prefijo simple.
    
    Args:
        prefix_string: Prefijo a probar (ej: "ABC")
    """
    print("=" * 60)
    print("🧪 PROBANDO MODELO StateEncoder")
    print("=" * 60)
    
    # 1. Cargar vocabulario
    vocab_path = Path(__file__).parent.parent / "vocab_char_to_id.json"
    if not vocab_path.exists():
        print(f"❌ No se encuentra: {vocab_path}")
        return
    
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    print(f"\n📚 Vocabulario cargado: {len(vocab)} tokens")
    print(f"   Ejemplos: {dict(list(vocab.items())[:5])}")
    
    # 2. Cargar modelo
    base_dir = Path(__file__).parent.parent
    model_path = base_dir / "checkpoints/state_encoder.pth"
    hparams_path = base_dir / "checkpoints/statenet_hparams.json"
    
    if not model_path.exists() or not hparams_path.exists():
        print(f"❌ No se encuentran los checkpoints:")
        print(f"   - {model_path}: {'✓' if model_path.exists() else '✗'}")
        print(f"   - {hparams_path}: {'✓' if hparams_path.exists() else '✗'}")
        return
    
    print(f"\n🤖 Cargando modelo...")
    if USE_LOCAL:
        encoder = load_state_encoder(model_path, hparams_path)
        device = encoder.device
        max_len = encoder.hparams["max_len"]
        d_state = encoder.hparams["d_state"]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder, hparams = load_model(str(model_path), str(hparams_path), device)
        max_len = hparams["max_len"]
        d_state = hparams["d_state"]
    
    print(f"   ✓ Modelo cargado en {device}")
    print(f"   - Max len: {max_len}")
    print(f"   - d_state: {d_state}")
    
    # 3. Convertir prefijo a IDs
    print(f"\n📥 ENTRADA:")
    print(f"   Prefijo: '{prefix_string}'")
    
    prefix_ids, length = string_to_ids(prefix_string, vocab, max_len)
    print(f"   IDs: {prefix_ids}")
    print(f"   Longitud: {length}")
    
    # 4. Crear tensores
    prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long)  # [1, seq_len]
    lengths_tensor = torch.tensor([length], dtype=torch.long)     # [1]
    
    # 5. Generar embedding
    print(f"\n🔄 Procesando...")
    if USE_LOCAL:
        embedding = encoder.encode(prefix_tensor, lengths_tensor)
    else:
        with torch.no_grad():
            embedding = encoder(prefix_tensor, lengths_tensor)
    
    # 6. Mostrar salida
    print(f"\n📤 SALIDA:")
    print(f"   Shape: {embedding.shape}")
    print(f"   Tipo: {type(embedding)}")
    print(f"   Primeros 10 valores: {embedding[0][:10].tolist()}")
    print(f"   Min: {embedding.min().item():.4f}")
    print(f"   Max: {embedding.max().item():.4f}")
    print(f"   Media: {embedding.mean().item():.4f}")
    
    # 7. Probar con varios prefijos
    print(f"\n" + "=" * 60)
    print("🧪 PROBANDO CON VARIOS PREFIJOS")
    print("=" * 60)
    
    test_prefixes = ["", "A", "AB", "ABC", "ABCD"]
    
    for prefix in test_prefixes:
        prefix_ids, length = string_to_ids(prefix, vocab, max_len)
        
        # Evitar length 0 (causa error en pack_padded_sequence)
        if length == 0:
            # Usar <EPS> para prefijo vacío
            prefix_ids = [vocab.get("<EPS>", 1)]
            length = 1
        
        prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long)
        lengths_tensor = torch.tensor([length], dtype=torch.long)
        
        try:
            if USE_LOCAL:
                emb = encoder.encode(prefix_tensor, lengths_tensor)
            else:
                with torch.no_grad():
                    emb = encoder(prefix_tensor, lengths_tensor)
            
            print(f"\n   Prefijo: '{prefix}' (len={length})")
            print(f"   → Embedding: shape {emb.shape}, norm={emb.norm().item():.4f}")
        except Exception as e:
            print(f"\n   Prefijo: '{prefix}' (len={length})")
            print(f"   → Error: {type(e).__name__}: {e}")
    
    print("\n" + "=" * 60)
    print("✅ PRUEBA COMPLETADA")
    print("=" * 60)
    print("\n💡 Interpretación:")
    print("   - Cada prefijo genera un vector de 128 números")
    print("   - Este vector representa el 'estado' del autómata")
    print("   - Prefijos diferentes → vectores diferentes")
    print("   - Prefijos similares → vectores similares (idealmente)")


if __name__ == "__main__":
    # Probar con prefijo por defecto
    test_model_simple("ABC")
    
    # O probar con otro prefijo
    # test_model_simple("XYZ")

