#!/usr/bin/env python3
"""
Script para probar los estados discretos generados.
Simula el autómata con cadenas y verifica aceptación.
"""

import torch
import numpy as np
from pathlib import Path


def load_states(states_path=None):
    """Carga los estados discretos generados."""
    if states_path is None:
        base_dir = Path(__file__).parent.parent
        states_path = base_dir / "artifacts/statenet/states_for_acceptnet.pt"
    
    path = Path(states_path)
    if not path.exists():
        print(f"❌ No se encuentra: {path}")
        return None
    
    print(f"📦 Cargando estados desde: {path}")
    states = torch.load(path, map_location='cpu', weights_only=False)
    print(f"   ✓ {len(states)} DFAs cargados")
    return states


def simulate_automaton(states, dfa_id, string, vocab):
    """
    Simula el autómata con una cadena.
    
    Args:
        states: Diccionario con estados de todos los DFAs
        dfa_id: ID del DFA a usar
        string: Cadena a probar (ej: "ABC")
        vocab: Vocabulario para convertir caracteres a índices
    
    Returns:
        (accepted, path): Si es aceptada y el camino de estados
    """
    if dfa_id not in states:
        print(f"❌ DFA {dfa_id} no encontrado")
        return False, []
    
    dfa = states[dfa_id]
    
    # Convertir a numpy si es tensor
    E = dfa["E"]
    m_use = dfa["m_use"]
    m_accept = dfa["m_accept"]
    delta = dfa["delta"]
    
    if isinstance(E, torch.Tensor):
        E = E.numpy()
    if isinstance(m_use, torch.Tensor):
        m_use = m_use.numpy()
    if isinstance(m_accept, torch.Tensor):
        m_accept = m_accept.numpy()
    if isinstance(delta, torch.Tensor):
        delta = delta.numpy()
    
    # Mapeo de símbolos a índices (A=0, B=1, ..., L=11, <EOS>=12)
    symbol_to_idx = {
        "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5,
        "G": 6, "H": 7, "I": 8, "J": 9, "K": 10, "L": 11
    }
    
    # Estado inicial (primer estado usado)
    initial_states = np.where(m_use)[0]
    if len(initial_states) == 0:
        print(f"   ⚠️  No hay estados usados en DFA {dfa_id}")
        return False, []
    
    current_state = initial_states[0]  # Empezar en el primer estado usado
    path = [current_state]
    
    # Simular transiciones
    for char in string:
        if char not in symbol_to_idx:
            print(f"   ⚠️  Caracter '{char}' no válido, saltando")
            continue
        
        symbol_idx = symbol_to_idx[char]
        
        # Verificar transición
        if current_state >= len(delta) or symbol_idx >= delta.shape[1]:
            print(f"   ⚠️  Índice fuera de rango: state={current_state}, symbol={symbol_idx}")
            return False, path
        
        next_state = delta[current_state, symbol_idx]
        
        # -1 significa transición inválida
        if next_state < 0:
            print(f"   ✗ Transición inválida desde estado {current_state} con '{char}'")
            return False, path
        
        # Verificar que el estado siguiente está usado
        if next_state >= len(m_use) or not m_use[next_state]:
            print(f"   ✗ Estado {next_state} no está usado")
            return False, path
        
        current_state = next_state
        path.append(current_state)
    
    # Verificar si el estado final es de aceptación
    is_accepting = m_accept[current_state] if current_state < len(m_accept) else False
    
    return is_accepting, path


def test_states_simple():
    """Prueba los estados discretos con cadenas simples."""
    print("=" * 60)
    print("🧪 PROBANDO ESTADOS DISCRETOS")
    print("=" * 60)
    
    # 1. Cargar estados
    states = load_states()
    if states is None:
        return
    
    # 2. Mostrar información de los DFAs disponibles
    print(f"\n📊 DFAs disponibles: {sorted(states.keys())}")
    
    # Mostrar info del primer DFA
    if len(states) > 0:
        first_dfa_id = sorted(states.keys())[0]
        dfa = states[first_dfa_id]
        
        # Convertir a numpy si es necesario
        m_use = dfa["m_use"]
        m_accept = dfa["m_accept"]
        delta = dfa["delta"]
        
        if isinstance(m_use, torch.Tensor):
            m_use = m_use.numpy()
        if isinstance(m_accept, torch.Tensor):
            m_accept = m_accept.numpy()
        if isinstance(delta, torch.Tensor):
            delta = delta.numpy()
        
        print(f"\n📋 Información del DFA {first_dfa_id}:")
        print(f"   - Estados usados: {m_use.sum()}/{len(m_use)}")
        print(f"   - Estados de aceptación: {m_accept.sum()}")
        print(f"   - Shape de transiciones: {delta.shape}")
    
    # 3. Probar con cadenas
    print(f"\n" + "=" * 60)
    print("🧪 SIMULANDO AUTÓMATAS CON CADENAS")
    print("=" * 60)
    
    # Vocabulario simple (solo para referencia)
    vocab = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5,
             "G": 6, "H": 7, "I": 8, "J": 9, "K": 10, "L": 11}
    
    # Probar con el primer DFA disponible
    if len(states) > 0:
        test_dfa_id = sorted(states.keys())[0]
        test_strings = ["A", "AB", "ABC", "ABCD", "XYZ"]
        
        print(f"\n   Probando DFA {test_dfa_id}:")
        for string in test_strings:
            accepted, path = simulate_automaton(states, test_dfa_id, string, vocab)
            status = "✓ ACEPTADA" if accepted else "✗ RECHAZADA"
            print(f"   '{string}' → {status} (camino: {path})")
    
    print("\n" + "=" * 60)
    print("✅ PRUEBA COMPLETADA")
    print("=" * 60)
    print("\n💡 Interpretación:")
    print("   - Los estados discretos representan el autómata completo")
    print("   - Puedes simular cadenas y ver si son aceptadas")
    print("   - El camino muestra qué estados se visitan")


if __name__ == "__main__":
    test_states_simple()

