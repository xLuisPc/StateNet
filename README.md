# StateNet

Modelo StateEncoder para codificar prefijos de cadenas en estados de DFA.

## Estructura

```
StateNet/
├── src/                    # Código fuente principal
│   ├── __init__.py
│   ├── model.py            # Arquitectura StateEncoder
│   ├── dataset.py          # Dataset y DataLoader
│   ├── train.py            # Script de entrenamiento local
│   └── infer.py            # Función de inferencia
├── scripts/                # Scripts de utilidad
│   ├── build_prefix_dataset.py  # Generar dataset de prefijos
│   └── colab/              # Scripts para Google Colab
│       ├── train_colab.py       # Entrenamiento en Colab
│       └── statenet_colab.py    # Pipeline completo en Colab
├── tests/                  # Scripts de prueba y evaluación
│   ├── test_model.py      # Probar modelo entrenado
│   ├── test_states.py     # Probar estados discretos
│   └── evaluate_model.py  # Evaluar modelo con datos reales
├── data/                   # Datos
│   ├── dataset6000.csv
│   └── statenet/
│       ├── prefix_train_sample.csv
│       ├── prefix_val_sample.csv
│       ├── prefix_test_sample.csv
│       └── meta.json
├── checkpoints/            # (se crea al entrenar)
│   ├── state_encoder.pth
│   ├── statenet_hparams.json
│   └── train_log.csv
├── artifacts/              # (se crea al construir estados)
│   └── statenet/
│       └── states_for_acceptnet.pt
├── vocab_char_to_id.json
└── requirements.txt
```

## Instalación

```bash
# Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

## Entrada y Salida del Modelo

### StateEncoder

**Entrada:**
- Prefijo: una secuencia de caracteres (ej: `"ABC"`)
- Se convierte a IDs usando el vocabulario
- Longitud del prefijo

**Salida:**
- Embedding del estado: vector de 128 números (d_state)
- Representa el estado del autómata después de procesar el prefijo

### Estados Discretos

**Entrada:**
- Cadena completa (ej: `"ABC"`)
- ID del DFA

**Salida:**
- Si la cadena es aceptada o rechazada
- Camino de estados visitados

## Uso

### 1. Generar dataset de prefijos

```bash
python scripts/build_prefix_dataset.py
```

Esto genera los archivos CSV en `data/statenet/`.

### 2. Entrenar modelo

El script `scripts/train.py` funciona tanto en local como en Colab, detectando automáticamente el entorno.

#### Entrenamiento (Local o Colab)

```bash
# En local
python scripts/train.py \
    --emb-dim 128 \
    --hidden-dim 256 \
    --d-state 128 \
    --batch-size 64 \
    --lr 1e-3 \
    --lambda-label 1.0 \
    --max-epochs 50 \
    --patience 5
```

O en Colab:

```python
# En Colab
exec(open('/content/StateNet/scripts/train.py').read())
```

El script detecta automáticamente si está en Colab (`/content` existe) o en local y ajusta las rutas.

**Archivos necesarios:**
- `vocab_char_to_id.json` - Vocabulario
- `data/statenet/prefix_train_sample.csv` - Dataset de entrenamiento (muestra)
- `data/statenet/prefix_val_sample.csv` - Dataset de validación (muestra)
- `data/statenet/meta.json` - Metadatos

### 3. Probar el modelo entrenado

```bash
python tests/test_model.py
```

Esto muestra:
- Entrada: prefijos como `"ABC"`
- Salida: embeddings (vectores de 128 números)
- Comparación entre diferentes prefijos

### 3b. Evaluar modelo con datos reales

```bash
python tests/evaluate_model.py
```

Evalúa el modelo con muestras de los datasets reales (train/val/test) y muestra métricas de rendimiento.

### 4. Generar estados discretos

Después de entrenar, genera embeddings, discretiza con k-means, construye transiciones y marca estados de aceptación:

```bash
# En local
python scripts/generate_states.py
```

O en Colab:

```python
# En Colab
exec(open('/content/StateNet/scripts/generate_states.py').read())
```

O importa y ejecuta:

```python
from scripts.generate_states import run_pipeline, CONFIG

run_pipeline(
    model_path="/content/StateNet/checkpoints/state_encoder.pth",
    hparams_path="/content/StateNet/checkpoints/statenet_hparams.json",
    vocab_path="/content/StateNet/vocab_char_to_id.json",
    train_csv_path="/content/StateNet/data/statenet/prefix_train_sample.csv",
    val_csv_path="/content/StateNet/data/statenet/prefix_val_sample.csv",
    output_path="/content/StateNet/artifacts/statenet/states_for_acceptnet.pt",
    test_csv_path="/content/StateNet/data/statenet/prefix_test_sample.csv",  # Opcional
    config=CONFIG
)
```

Esto genera:
- `artifacts/statenet/states_for_acceptnet.pt`: Autómatas discretos completos con:
  - `E`: Centroides de estados [K_max, d_state]
  - `m_use`: Estados usados [K_max] bool
  - `m_accept`: Estados de aceptación [K_max] bool
  - `delta`: Tabla de transiciones [K_max, num_symbols]

### 5. Probar estados discretos

```bash
python tests/test_states.py
```

Esto simula el autómata con cadenas y muestra si son aceptadas o rechazadas.

## Inferencia programática

### Usar el modelo directamente

```python
from src.infer import load_state_encoder
import torch

# Cargar modelo
encoder = load_state_encoder(
    "checkpoints/state_encoder.pth",
    "checkpoints/statenet_hparams.json"
)

# Codificar prefijos
prefix_ids = torch.tensor([[2, 3, 4]], dtype=torch.long)  # "ABC" -> [A=2, B=3, C=4]
lengths = torch.tensor([3], dtype=torch.long)

h_t = encoder.encode(prefix_ids, lengths)  # [batch, d_state]
print(h_t.shape)  # torch.Size([1, 128])
```

### Usar estados discretos

```python
import torch

# Cargar estados
states = torch.load("artifacts/statenet/states_for_acceptnet.pt", map_location='cpu')

# Obtener estados de un DFA
dfa_id = 0
dfa = states[dfa_id]

E = dfa["E"]          # Centroides [K_max, d_state]
m_use = dfa["m_use"]   # Estados usados [K_max] bool
m_accept = dfa["m_accept"]  # Estados aceptación [K_max] bool
delta = dfa["delta"]   # Transiciones [K_max, num_symbols]
```

## Arquitectura

### StateEncoder

- **Entrada**: `prefix_ids` [batch, seq_len], `lengths` [batch]
- **Embedding**: Tokens → `emb_dim`
- **RNN**: GRU/LSTM unidireccional → `hidden_dim`
- **Estado final**: Último estado válido por secuencia
- **Salida**: `h_t` [batch, `d_state`]

### StateEncoderWithHeads

Extiende StateEncoder con dos cabezas de predicción:
- **next_symbol**: Clasificación sobre símbolos (A-L + <EOS>)
- **final_label**: Clasificación binaria (BCE)

### Loss

```
Loss = CE(next_symbol) + λ * BCE(final_label)
```

Donde `λ` (lambda-label) controla el peso relativo (default: 1.0).

## Flujo completo

1. **Prefijo** → Modelo → **Embedding** (vector de 128 números)
2. Muchos embeddings → k-means → **Estados discretos**
3. Estados discretos → **Autómata completo**
4. Cadena → Autómata → **Aceptada/Rechazada**

## Archivos generados

### Entrenamiento
- `checkpoints/state_encoder.pth`: Pesos del encoder
- `checkpoints/statenet_hparams.json`: Hiperparámetros del modelo
- `checkpoints/train_log.csv`: Log de entrenamiento

### Construcción de estados
- `artifacts/statenet/states_for_acceptnet.pt`: Autómatas discretos completos
