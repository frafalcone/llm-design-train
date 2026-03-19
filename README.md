# LLM from Scratch — README

> 🇮🇹 [Italiano](#italiano) · 🇬🇧 [English](#english)

---

<a name="italiano"></a>
## Italiano

### Panoramica

Implementazione di un **Large Language Model decoder-only** scritto da zero in PyTorch, ispirato all'architettura LLaMA. Il modello è addestrato sul task di **causal language modeling** (predizione del token successivo).

### Architettura del Modello

| Componente | Tecnica |
|---|---|
| Tipo | Transformer Decoder-only |
| Normalizzazione | RMSNorm (pre-norm) |
| Attenzione | Grouped Query Attention (GQA) |
| Embedding posizionale | Rotary Position Embedding (RoPE) |
| Funzione di attivazione | SwiGLU |
| Tokenizer | GPT-2 BPE (`tiktoken`) |
| Dimensione embedding | 768 |
| Layer | 12 |
| Teste Q / Gruppi K-V | 12 / 4 |
| Lunghezza contesto | 1024 token |
| Vocabolario | 50.257 token |

**Weight tying:** i pesi dell'embedding di input e dell'output head sono condivisi, riducendo i parametri e migliorando la generalizzazione.

**Inizializzazione pesi:** `trunc_normal_(std=0.02)`; i layer nelle connessioni residuali usano `std *= (2 * num_layers)^-0.5` (tecnica GPT-2).

### Pipeline dei Dati

- **Formato input:** file Parquet con colonna `text`
- **Tokenizzazione:** GPT-2 BPE via `tiktoken`, salvata in file `.bin` come array `int32`
- **Lettura dataset:** `numpy.memmap` — nessun caricamento in RAM
- **Sequenze:** input `x` e target `y` sfasati di un token

```
x = tokens[start : end]
y = tokens[start+1 : end+1]
```

### Training

| Parametro | Valore |
|---|---|
| Ottimizzatore | AdamW (`β=(0.9, 0.95)`, `wd=0.2`) |
| Learning rate | `1e-3` con cosine decay + warmup |
| Precisione mista | `bfloat16` (Ampere+) o `float16` + GradScaler |
| Gradient accumulation | 32 step → batch effettivo 256 |
| Gradient clipping | `max_norm=1.0` |
| Epoche | 3 |

Il checkpoint viene salvato ogni ~20% di un'epoca e al termine di ciascuna. Il modello con la miglior validation loss è salvato separatamente in `output/bckpt.pth`. Il resume salta i batch già processati tramite `itertools.islice`, senza caricarli su GPU. Il gradient checkpointing è supportato per ridurre il consumo di VRAM.

### Struttura del Progetto

```
project/
│
├── src/
│   ├── architecture/
│   │   ├── configuration.py           # Iperparametri del modello
│   │   ├── model.py                   # Classe principale Model
│   │   ├── transformer_decoder.py     # Blocco Transformer Decoder
│   │   ├── grouped_query_attention.py # GQA + RoPE
│   │   ├── feedforward.py             # FFN con SwiGLU
│   │   └── rmsnorm.py                 # Root Mean Square Normalization
│   │
│   ├── training/
│   │   ├── configuration.py           # Iperparametri di training
│   │   └── train.py                   # Loop di training e validazione
│   │
│   ├── dataengine/
│   │   ├── configuration.py           # Configurazione dati
│   │   ├── dataset.py                 # CustomDataset + DataLoader
│   │   └── preprocess_data.py         # Tokenizzazione Parquet → .bin
│   │
│   ├── utils/
│   │   ├── state_manager.py           # Salvataggio/caricamento checkpoint completo
│   │   └── model_manager.py           # Salvataggio/caricamento solo pesi
│   │
│   └── main.py                        # Entry point
│
├── data/
│   ├── parq/                          # Dataset grezzo (Parquet)
│   └── bin/                           # Dataset tokenizzato (binario)
│
├── output/
│   ├── lckpt.pth                      # Ultimo checkpoint
│   ├── bckpt.pth                      # Miglior checkpoint (val loss)
│   ├── model.pth                      # Modello finale (solo pesi)
│   └── training_results.json          # Metriche finali
│
└── docs/                              # Documentazione e Benchmark
```

### Requisiti

```
torch >= 2.0
numpy
pandas
tiktoken
tqdm
pyarrow
```

### Avvio

```bash
python src/main.py
```

Il programma tokenizza automaticamente i dati (se i file `.bin` non esistono), avvia il training e salva il modello finale in `output/model.pth`.

---

<a name="english"></a>
## English

### Overview

A **decoder-only Large Language Model** implemented from scratch in PyTorch, inspired by the LLaMA family architecture. The model is trained on the **causal language modeling** task (next-token prediction).

### Model Architecture

| Component | Technique |
|---|---|
| Type | Decoder-only Transformer |
| Normalization | RMSNorm (pre-norm) |
| Attention | Grouped Query Attention (GQA) |
| Positional Embedding | Rotary Position Embedding (RoPE) |
| Activation Function | SwiGLU |
| Tokenizer | GPT-2 BPE (`tiktoken`) |
| Embedding Size | 768 |
| Layers | 12 |
| Q Heads / K-V Groups | 12 / 4 |
| Context Length | 1024 tokens |
| Vocabulary | 50,257 tokens |

**Weight tying:** input embedding and output head share the same weights, reducing parameter count and improving generalization.

**Weight initialization:** `trunc_normal_(std=0.02)`; residual projection layers use `std *= (2 * num_layers)^-0.5` (GPT-2 technique).

### Data Pipeline

- **Input format:** Parquet files with a `text` column
- **Tokenization:** GPT-2 BPE via `tiktoken`, saved to `.bin` files as `int32` arrays
- **Dataset reading:** `numpy.memmap` — no full RAM load required
- **Sequences:** input `x` and target `y` are offset by one token

```
x = tokens[start : end]
y = tokens[start+1 : end+1]
```

### Training

| Parameter | Value |
|---|---|
| Optimizer | AdamW (`β=(0.9, 0.95)`, `wd=0.2`) |
| Learning rate | `1e-3` with cosine decay + warmup |
| Mixed precision | `bfloat16` (Ampere+) or `float16` + GradScaler |
| Gradient accumulation | 32 steps → effective batch size 256 |
| Gradient clipping | `max_norm=1.0` |
| Epochs | 3 |

A checkpoint is saved approximately every 20% of an epoch and unconditionally at the end of each one. The best model by validation loss is saved separately to `output/bckpt.pth`. Resume skips already-processed batches via `itertools.islice`, without loading them onto the GPU. Gradient checkpointing is supported to reduce VRAM usage.

### Project Structure

```
project/
│
├── src/
│   ├── architecture/
│   │   ├── configuration.py           # Model hyperparameters
│   │   ├── model.py                   # Main Model class
│   │   ├── transformer_decoder.py     # Transformer Decoder block
│   │   ├── grouped_query_attention.py # GQA + RoPE
│   │   ├── feedforward.py             # FFN with SwiGLU
│   │   └── rmsnorm.py                 # Root Mean Square Normalization
│   │
│   ├── training/
│   │   ├── configuration.py           # Training hyperparameters
│   │   └── train.py                   # Training and validation loop
│   │
│   ├── dataengine/
│   │   ├── configuration.py           # Data configuration
│   │   ├── dataset.py                 # CustomDataset + DataLoader
│   │   └── preprocess_data.py         # Parquet → .bin tokenization
│   │
│   ├── utils/
│   │   ├── state_manager.py           # Full checkpoint save/load
│   │   └── model_manager.py           # Weights-only save/load
│   │
│   └── main.py                        # Entry point
│
├── data/
│   ├── parq/                          # Raw dataset in Parquet format
│   └── bin/                           # Tokenized dataset in binary format
│
├── output/
│   ├── lckpt.pth                      # Latest checkpoint
│   ├── bckpt.pth                      # Best checkpoint (val loss)
│   ├── model.pth                      # Final model (weights only)
│   └── training_results.json          # Final metrics
│
└── docs/                              # Documentation and Benchmark
```

### Requirements

```
torch >= 2.0
numpy
pandas
tiktoken
tqdm
pyarrow
```

### Quick Start

```bash
python src/main.py
```

The program automatically tokenizes the data (if `.bin` files do not already exist), runs the training loop, and saves the final model to `output/model.pth`.