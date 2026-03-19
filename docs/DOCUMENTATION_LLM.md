# Project Documentation — LLM from Scratch

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Model Architecture](#2-model-architecture)
   - [Configuration](#21-configuration)
   - [Embedding and Output Head](#22-embedding-and-output-head)
   - [RMSNorm](#23-rmsnorm)
   - [Grouped Query Attention (GQA)](#24-grouped-query-attention-gqa)
   - [FeedForward (SwiGLU)](#25-feedforward-swiglu)
   - [Transformer Decoder Block](#26-transformer-decoder-block)
   - [Model](#27-model)
3. [Data Pipeline](#3-data-pipeline)
   - [Data Configuration](#31-data-configuration)
   - [Preprocessing and Tokenization](#32-preprocessing-and-tokenization)
   - [Dataset and DataLoader](#33-dataset-and-dataloader)
4. [Training](#4-training)
   - [Training Configuration](#41-training-configuration)
   - [Training Loop](#42-training-loop)
   - [Validation](#43-validation)
5. [State and Checkpoint Management](#5-state-and-checkpoint-management)
6. [Entry Point — main.py](#6-entry-point--mainpy)
7. [Project Structure](#7-project-structure)
8. [Requirements](#8-requirements)

---

## 1. Project Overview

This project implements a **decoder-only Large Language Model (LLM) from scratch** in PyTorch, inspired by the LLaMA family architecture. The model is designed for the **causal language modeling** task (next-token prediction) and includes all the modern components of a high-performance transformer.

| Feature | Value / Technique |
|---|---|
| Architecture | Decoder-only Transformer |
| Normalization | RMSNorm (pre-norm) |
| Attention | Grouped Query Attention (GQA) |
| Positional Embedding | Rotary Position Embedding (RoPE) |
| Activation Function | SwiGLU |
| Tokenizer | GPT-2 BPE (via `tiktoken`) |
| Embedding Size | 768 |
| Number of Layers | 12 |
| Number of Heads | 12 (Q) / 4 groups (K, V) — 3:1 query-per-K/V ratio |
| Context Length | 1024 tokens |
| Vocabulary | 50,257 tokens |

---

## 2. Model Architecture

### 2.1 Configuration

**File:** `architecture/configuration.py`

Defines all model hyperparameters via Python dictionaries passed to individual modules. The main parameters are:

```python
embedding = 768               # Embedding space dimension
number_of_heads = 12          # Number of query heads
number_of_groups = 4          # Number of K/V groups (GQA)
context_length = 1024         # Maximum sequence length
layers = 12                   # Number of Transformer Decoder blocks
vocabulary = 50257            # GPT-2 vocabulary size
dropout_rate = 0.1            # Dropout probability
bias = False                  # No bias in linear layers
epsilon = 1e-6                # Epsilon for numerical stability in RMSNorm
embedding_expansion_rate = 4  # Expansion factor for the FFN
```

The configuration is organized hierarchically: `model_configuration` contains `trf_configuration`, which in turn contains `gqa_configuration`, `ffn_configuration`, and `rmsn_configuration`.

---

### 2.2 Embedding and Output Head

**File:** `architecture/model.py`

The model uses **weight tying**: the weights of the input embedding (`tok_emb`) and the output head (`out_head`) are shared. This technique reduces the number of parameters and improves generalization.

```python
self.tok_emb  = torch.nn.Embedding(vocabulary, embedding)
self.out_head = torch.nn.Linear(embedding, vocabulary, bias)
self.out_head.weight = self.tok_emb.weight  # weight tying
```

---

### 2.3 RMSNorm

**File:** `architecture/rmsnorm.py`

Implements **Root Mean Square Layer Normalization**, which is more efficient than standard LayerNorm because it removes the mean computation (re-centering). Normalization is performed in `float32` for numerical stability; the result is then cast back to the original input dtype before being multiplied by the learnable parameter γ, initialized to 1.

**Formula:**

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \varepsilon}} \cdot \gamma$$

**Pre-norm scheme:** normalization is applied **before** the attention layer and the FFN — a pattern adopted by modern architectures (LLaMA, Mistral) for greater training stability.

---

### 2.4 Grouped Query Attention (GQA)

**File:** `architecture/grouped_query_attention.py`

Implements **Grouped Query Attention**, an optimization over standard Multi-Head Attention. The Query projections have `num_heads = 12` heads, while the Key and Value projections have only `num_groups = 4` heads (one group per 3 query heads). This significantly reduces the memory footprint of the KV-cache during inference.

**Linear projections:**

```python
q_proj:   Linear(768 → 768)   # 12 heads × 64 head_dim
k_proj:   Linear(768 → 256)   # 4 groups × 64 head_dim
v_proj:   Linear(768 → 256)   # 4 groups × 64 head_dim
out_proj: Linear(768 → 768)
```

**Rotary Position Embedding (RoPE):** the `cos` and `sin` coefficients are precomputed for all positions up to `context_length` and saved as buffers (not parameters). They are applied directly to the Q and K projections via rotation in the real-valued formulation with even/odd index separation, without being added to the embeddings as in the original GPT.

**Scaled Dot-Product Attention:** uses `torch.nn.functional.scaled_dot_product_attention` with `is_causal=True`, which internally leverages Flash Attention when available on compatible hardware.

---

### 2.5 FeedForward (SwiGLU)

**File:** `architecture/feedforward.py`

Implements a feed-forward network with **SwiGLU** activation, adopted by LLaMA as an alternative to the classic FFN with ReLU or GELU.

**Formula:**

$$\text{FFN}(x) = \text{SiLU}(\text{gate\_proj}(x)) \cdot \text{up\_proj}(x) \cdot W_{down}$$

The hidden dimension is computed as:

```python
hidden_dim = int(embedding * expansion_rate * 2 / 3)
hidden_dim = 8 * ((hidden_dim + 8 - 1) // 8)  # round up to multiple of 8

# With default values (embedding=768, expansion_rate=4):
# hidden_dim = int(768 * 4 * 2/3) = 2048  →  already a multiple of 8  →  hidden_dim = 2048
```

---

### 2.6 Transformer Decoder Block

**File:** `architecture/transformer_decoder.py`

Each block follows the **pre-norm** scheme with residual connections, implemented in compact form:

```
x  →  x + Dropout(GQA(Norm1(x)))
x  →  x + Dropout(FFN(Norm2(x)))
```

Gradient checkpointing is optionally supported: when enabled via `model.gradient_checkpointing_enable()`, each block recomputes activations during the backward pass instead of keeping them in memory, reducing VRAM usage at the cost of additional computation.

No cross-attention is present: this is a pure causal decoder (GPT-style), not an encoder-decoder like the original Vaswani et al. Transformer.

---

### 2.7 Model

**File:** `architecture/model.py`

The complete model is composed of:

| # | Component | Description |
|---|---|---|
| 1 | Token Embedding | Maps token indices to embedding vectors |
| 2 | Dropout | Applied to input embeddings |
| 3 | Stack × 12 TransformerDecoder | Sequential blocks via `nn.Sequential` |
| 4 | Final RMSNorm | Normalizes the output of the entire stack |
| 5 | Output Head | Linear projection to vocabulary (weight tying) |

**Weight initialization:** linear weights are initialized with `trunc_normal_(std=0.02)`. Layers that are part of residual connections (marked with `is_residual_proj=True`) use a std scaled by model depth:

```python
std *= (2 * num_layers) ** -0.5
```

This technique (inspired by GPT-2) prevents gradient explosion in deep networks.

---

## 3. Data Pipeline

### 3.1 Data Configuration

**File:** `dataengine/configuration.py`

```python
context_size = 1024   # Tokens per sequence
batch        = 8      # Batch size
num_workers  = 4      # DataLoader workers
percentage   = 1.0    # Fraction of dataset to use (1.0 = all)
```

---

### 3.2 Preprocessing and Tokenization

**File:** `dataengine/preprocess_data.py`

Raw data in **Parquet** format (column `text`) is tokenized using the **GPT-2 BPE** tokenizer (`tiktoken`) and saved to binary `.bin` files as `int32` integer arrays. Each document is separated by the special token `<|endoftext|>`. The process is optimized via batch tokenization (`encode_ordinary_batch`) to reduce I/O overhead.

---

### 3.3 Dataset and DataLoader

**File:** `dataengine/dataset.py`

`CustomDataset` reads `.bin` files using **memory-mapped files** (`numpy.memmap`), avoiding loading the entire dataset into RAM. Input sequences `x` and targets `y` are offset by one token:

```python
x = tokens[start : end]      # input
y = tokens[start+1 : end+1]  # target (shifted by 1)
```

The `DataLoader` supports `pin_memory`, `prefetch_factor`, and `persistent_workers` to maximize throughput during GPU training. The `percentage` parameter can be passed directly to `create_dataloader` to use only a fraction of the dataset, useful for quick experiments; the value in the `data_configuration` dictionary is not read by the function.

---

## 4. Training

### 4.1 Training Configuration

**File:** `training/configuration.py`

```python
epochs             = 3            # Number of epochs
learning_rate      = 1e-3         # Initial learning rate
weight_decay       = 0.2          # L2 weight decay
betas              = (0.9, 0.95)  # AdamW beta parameters
lr_min_ratio       = 0.1          # Min LR = 10% of initial LR
accumulation_steps = 32           # Gradient accumulation steps
use_scheduler      = True         # Cosine scheduler with warmup enabled
warmup_ratio       = 0.1          # Percentage of warmup steps
```

---

### 4.2 Training Loop

**File:** `training/train.py`

#### Optimizer — AdamW with Decoupled Weight Decay

Parameters with dimensionality ≥ 2 (weight matrices) receive weight decay; others (biases, normalization parameters with `dim < 2`) do not. This follows standard recommended practice for transformers.

#### Mixed Precision

Training uses `torch.amp.autocast`. On CUDA GPUs, `bfloat16` is attempted first (supported from the Ampere architecture onward); otherwise it falls back to `float16`. Mixed precision is disabled on CPU. The `GradScaler` is active only with `float16` on CUDA — `bfloat16` does not require loss scaling.

#### Gradient Accumulation

Gradients are accumulated for `accumulation_steps = 32` batches before each optimizer step. With `batch_size = 8` (defined in `dataengine/configuration.py`, a separate and independent file), the effective batch size is **8 × 32 = 256 samples**.

#### Gradient Clipping

The gradient norm is clipped to `max_norm = 1.0`. If `grad_norm` results in `NaN` or `Inf`, the entire update block is skipped: no optimizer step, no scheduler update, ensuring training stability.

#### Learning Rate Scheduler — Cosine Decay with Warmup

Active when `use_scheduler = True`. The number of warmup steps is computed internally as `warmup_steps = int(total_optimization_steps × warmup_ratio)`, where `warmup_ratio` is configurable in `trn_configuration`. During warmup, the LR increases linearly from 0 to the initial value; it then decays via cosine annealing down to `lr_min_ratio × learning_rate`.

If `use_scheduler = False`, the scheduler applies a constant function λ = 1.0 and the **LR remains fixed** throughout training.

#### Progressive Saving

A checkpoint is saved approximately every 20% of an epoch's iterations (`save_step = max(1, total_batches // 5)`) and unconditionally at the end of each epoch. The model with the best validation loss is saved separately to `output/bckpt.pth`.

#### Training Resume

When restoring from a checkpoint, already-processed batches in the current epoch are skipped using `itertools.islice` on the DataLoader iterator. This avoids loading and transferring redundant data to the GPU, making resume efficient even on large datasets.

---

### 4.3 Validation

**File:** `training/train.py` — `validate_model()`

The validation loss is computed at the end of each epoch with the model in `eval()` mode and `torch.no_grad()`. The loss is weighted by the number of tokens (not batches) for a correct comparison across batches of different sizes. If the loss is `NaN` or `Inf`, the method returns `float('inf')`.

---

## 5. State and Checkpoint Management

**File:** `utils/state_manager.py`, `utils/model_manager.py`

`save_state` saves a complete checkpoint that allows training to be resumed exactly from any point:

| Field | Description |
|---|---|
| `model_state_dict` | Model weights |
| `optimizer_state_dict` | AdamW state (moment estimates) |
| `scheduler_state_dict` | LR scheduler state |
| `scaler_state_dict` | GradScaler state (AMP) |
| `epoch`, `batch` | Current position in the dataset |
| `current_opt_step` | Current optimization step |
| `rng_state`, `cuda_rng_state` | Random generator states (CPU and GPU) |

**Atomic saving:** the file is written first as `.tmp` and then renamed with `os.replace`, preventing corrupted checkpoints in case of sudden interruption.

**Note:** `load_model` (in `model_manager.py`) is a lightweight version that loads only the model weights, intended for inference. If the file does not exist, the function returns silently without raising exceptions.

---

## 6. Entry Point — main.py

**File:** `main.py`

Main project orchestrator. The execution flow is:

```
1. Device initialization (CUDA > CPU)
2. Model creation
3. Dataset tokenization (only if .bin files do not already exist)
4. DataLoader creation for training and validation
5. Training (train_model)
6. Final model saving — weights only — to output/model.pth
```

---

## 7. Project Structure

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
└── docs/                              # Documentation
```

---

## 8. Requirements

| Package | Minimum Version | Usage |
|---|---|---|
| `torch` | >= 2.0 | Main framework, AMP, SDPA / Flash Attention |
| `numpy` | any | Memmap reading and binary tokenization |
| `pandas` | any | Parquet file reading |
| `tiktoken` | any | GPT-2 BPE tokenizer |
| `tqdm` | any | Training progress bars |
| `pyarrow` | any | Backend engine for Parquet reading |

> **GPU Note:** for mixed precision training with `bfloat16`, a GPU with NVIDIA Ampere architecture or later is recommended (e.g., RTX 3000, A100). On older hardware, training automatically falls back to `float16` with GradScaler.