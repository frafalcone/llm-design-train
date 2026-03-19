# Documentazione del Progetto — LLM da Zero

## Indice

1. [Panoramica del Progetto](#1-panoramica-del-progetto)
2. [Architettura del Modello](#2-architettura-del-modello)
   - [Configurazione](#21-configurazione)
   - [Embedding e Output Head](#22-embedding-e-output-head)
   - [RMSNorm](#23-rmsnorm)
   - [Grouped Query Attention (GQA)](#24-grouped-query-attention-gqa)
   - [FeedForward (SwiGLU)](#25-feedforward-swiglu)
   - [Transformer Decoder Block](#26-transformer-decoder-block)
   - [Model](#27-model)
3. [Pipeline dei Dati](#3-pipeline-dei-dati)
   - [Configurazione Dati](#31-configurazione-dati)
   - [Pre-elaborazione e Tokenizzazione](#32-pre-elaborazione-e-tokenizzazione)
   - [Dataset e DataLoader](#33-dataset-e-dataloader)
4. [Training](#4-training)
   - [Configurazione Training](#41-configurazione-training)
   - [Loop di Training](#42-loop-di-training)
   - [Validazione](#43-validazione)
5. [Gestione degli State e dei Checkpoint](#5-gestione-degli-state-e-dei-checkpoint)
6. [Entry Point — main.py](#6-entry-point--mainpy)
7. [Struttura del Progetto](#7-struttura-del-progetto)
8. [Requisiti](#8-requisiti)

---

## 1. Panoramica del Progetto

Questo progetto implementa un **Large Language Model (LLM) decoder-only da zero** in PyTorch, ispirato all'architettura della famiglia LLaMA. Il modello è progettato per il task di **language modeling causale** (predizione del prossimo token) e include tutte le componenti moderne di un transformer ad alte prestazioni.

| Caratteristica | Valore / Tecnica |
|---|---|
| Architettura | Transformer Decoder-only |
| Normalizzazione | RMSNorm (pre-norm) |
| Attention | Grouped Query Attention (GQA) |
| Embedding posizionale | Rotary Position Embedding (RoPE) |
| Funzione di attivazione | SwiGLU |
| Tokenizzatore | GPT-2 BPE (via `tiktoken`) |
| Dimensione embedding | 768 |
| Numero di layer | 12 |
| Numero di heads | 12 (Q) / 4 gruppi (K, V) — rapporto 3:1 query per coppia K/V |
| Context length | 1024 token |
| Vocabolario | 50.257 token |

---

## 2. Architettura del Modello

### 2.1 Configurazione

**File:** `architecture/configuration.py`

Definisce tutti gli iperparametri del modello tramite dizionari Python passati ai singoli moduli. I parametri principali sono:

```python
embedding = 768               # Dimensione dello spazio di embedding
number_of_heads = 12          # Numero di head di query
number_of_groups = 4          # Numero di gruppi K/V (GQA)
context_length = 1024         # Lunghezza massima della sequenza
layers = 12                   # Numero di Transformer Decoder blocks
vocabulary = 50257            # Dimensione del vocabolario GPT-2
dropout_rate = 0.1            # Probabilità di dropout
bias = False                  # Nessun bias nei layer lineari
epsilon = 1e-6                # Epsilon per la stabilità numerica in RMSNorm
embedding_expansion_rate = 4  # Fattore di espansione per la FFN
```

La configurazione è organizzata gerarchicamente: `model_configuration` contiene `trf_configuration`, che a sua volta contiene `gqa_configuration`, `ffn_configuration` e `rmsn_configuration`.

---

### 2.2 Embedding e Output Head

**File:** `architecture/model.py`

Il modello utilizza **weight tying**: i pesi dell'embedding di input (`tok_emb`) e dell'output head (`out_head`) sono condivisi. Questa tecnica riduce il numero di parametri e migliora la generalizzazione.

```python
self.tok_emb  = torch.nn.Embedding(vocabulary, embedding)
self.out_head = torch.nn.Linear(embedding, vocabulary, bias)
self.out_head.weight = self.tok_emb.weight  # weight tying
```

---

### 2.3 RMSNorm

**File:** `architecture/rmsnorm.py`

Implementa la **Root Mean Square Layer Normalization**, più efficiente della LayerNorm classica perché elimina il calcolo della media (re-centering). La normalizzazione avviene in `float32` per stabilità numerica; il risultato viene poi riportato al dtype originale dell'input prima di essere moltiplicato per il parametro apprendibile γ, inizializzato a 1.

**Formula:**

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \varepsilon}} \cdot \gamma$$

**Schema pre-norm:** la normalizzazione viene applicata **prima** del layer di attention e della FFN, schema adottato dalle architetture moderne (LLaMA, Mistral) per una maggiore stabilità del training.

---

### 2.4 Grouped Query Attention (GQA)

**File:** `architecture/grouped_query_attention.py`

Implementa la **Grouped Query Attention**, un'ottimizzazione della Multi-Head Attention standard. Le proiezioni Query hanno `num_heads = 12` head, mentre le proiezioni Key e Value hanno solo `num_groups = 4` head (un gruppo ogni 3 head di query). Questo riduce significativamente la memoria occupata dalla KV-cache durante l'inferenza.

**Proiezioni lineari:**

```python
q_proj:   Linear(768 → 768)   # 12 heads × 64 head_dim
k_proj:   Linear(768 → 256)   # 4 gruppi × 64 head_dim
v_proj:   Linear(768 → 256)   # 4 gruppi × 64 head_dim
out_proj: Linear(768 → 768)
```

**Rotary Position Embedding (RoPE):** i coefficienti `cos` e `sin` sono pre-calcolati per tutte le posizioni fino a `context_length` e salvati come buffer (non come parametri). Vengono applicati direttamente alle proiezioni Q e K tramite rotazione nella formulazione reale con separazione degli indici pari/dispari, senza essere sommati agli embedding come nel GPT originale.

**Scaled Dot-Product Attention:** utilizza `torch.nn.functional.scaled_dot_product_attention` con `is_causal=True`, che sfrutta internamente Flash Attention quando disponibile su hardware compatibile.

---

### 2.5 FeedForward (SwiGLU)

**File:** `architecture/feedforward.py`

Implementa una rete feed-forward con attivazione **SwiGLU**, adottata da LLaMA in alternativa alla classica FFN con ReLU o GELU.

**Formula:**

$$\text{FFN}(x) = \text{SiLU}(\text{gate\_proj}(x)) \cdot \text{up\_proj}(x) \cdot W_{down}$$

La dimensione nascosta viene calcolata come:

```python
hidden_dim = int(embedding * expansion_rate * 2 / 3)
hidden_dim = 8 * ((hidden_dim + 8 - 1) // 8)  # arrotondamento al multiplo di 8

# Con i valori di default (embedding=768, expansion_rate=4):
# hidden_dim = int(768 * 4 * 2/3) = 2048  →  già multiplo di 8  →  hidden_dim = 2048
```

---

### 2.6 Transformer Decoder Block

**File:** `architecture/transformer_decoder.py`

Ogni blocco segue lo schema **pre-norm** con connessioni residuali, implementato nella forma compatta:

```
x  →  x + Dropout(GQA(Norm1(x)))
x  →  x + Dropout(FFN(Norm2(x)))
```

Il gradient checkpointing è supportato opzionalmente: se abilitato tramite `model.gradient_checkpointing_enable()`, ogni blocco ricalcola le attivazioni durante il backward invece di tenerle in memoria, riducendo il consumo di VRAM a fronte di un overhead computazionale.

Nessun cross-attention è presente: si tratta di un decoder causale puro (stile GPT), non di un encoder-decoder come il Transformer originale di Vaswani et al.

---

### 2.7 Model

**File:** `architecture/model.py`

Il modello completo è composto da:

| # | Componente | Descrizione |
|---|---|---|
| 1 | Token Embedding | Mappa gli indici dei token in vettori d'embedding |
| 2 | Dropout | Applicato agli embedding di input |
| 3 | Stack × 12 TransformerDecoder | Blocchi in sequenza tramite `nn.Sequential` |
| 4 | RMSNorm finale | Normalizzazione dell'output dell'intero stack |
| 5 | Output Head | Proiezione lineare verso il vocabolario (weight tying) |

**Inizializzazione dei pesi:** i pesi lineari sono inizializzati con `trunc_normal_(std=0.02)`. I layer che fanno parte delle connessioni residuali (marcati con `is_residual_proj=True`) usano una std scalata in funzione della profondità del modello:

```python
std *= (2 * num_layers) ** -0.5
```

Questa tecnica (ispirata a GPT-2) previene l'esplosione dei gradienti nelle reti profonde.

---

## 3. Pipeline dei Dati

### 3.1 Configurazione Dati

**File:** `dataengine/configuration.py`

```python
context_size = 1024   # Token per sequenza
batch        = 8      # Batch size
num_workers  = 4      # Worker per il DataLoader
percentage   = 1.0    # Frazione del dataset da usare (1.0 = tutto)
```

---

### 3.2 Pre-elaborazione e Tokenizzazione

**File:** `dataengine/preprocess_data.py`

I dati grezzi in formato **Parquet** (colonna `text`) vengono tokenizzati con il tokenizzatore **GPT-2 BPE** (`tiktoken`) e salvati in file `.bin` binari come array di interi `int32`. Ogni documento è separato dal token speciale `<|endoftext|>`. Il processo è ottimizzato tramite tokenizzazione in batch (`encode_ordinary_batch`) per ridurre l'overhead di I/O.

---

### 3.3 Dataset e DataLoader

**File:** `dataengine/dataset.py`

`CustomDataset` legge i file `.bin` tramite **memory-mapped file** (`numpy.memmap`), evitando di caricare l'intero dataset in RAM. Le sequenze di input `x` e target `y` sono slittate di un token:

```python
x = tokens[start : end]      # input
y = tokens[start+1 : end+1]  # target (shifted by 1)
```

Il `DataLoader` supporta `pin_memory`, `prefetch_factor` e `persistent_workers` per massimizzare il throughput durante il training su GPU. Il parametro `percentage` può essere passato direttamente a `create_dataloader` per usare solo una frazione del dataset, utile per esperimenti rapidi; il valore nel dizionario `data_configuration` non viene letto dalla funzione.

---

## 4. Training

### 4.1 Configurazione Training

**File:** `training/configuration.py`

```python
epochs             = 3            # Numero di epoche
learning_rate      = 1e-3         # Learning rate iniziale
weight_decay       = 0.2          # Peso del decadimento L2
betas              = (0.9, 0.95)  # Parametri beta di AdamW
lr_min_ratio       = 0.1          # LR minimo = 10% del LR iniziale
accumulation_steps = 32           # Step di gradient accumulation
use_scheduler      = True         # Cosine scheduler con warmup abilitato
warmup_ratio       = 0.1          # Percentuale di step di warmup
```

---

### 4.2 Loop di Training

**File:** `training/train.py`

#### Ottimizzatore — AdamW con decoupled weight decay

I parametri con dimensionalità ≥ 2 (matrici di peso) ricevono weight decay; gli altri (bias, parametri di normalizzazione con `dim < 2`) no. Questo segue la pratica standard raccomandata per i transformer.

#### Mixed Precision

Il training usa `torch.amp.autocast`. Su GPU CUDA si tenta prima `bfloat16` (supportato da architettura Ampere in poi); in caso contrario si ricade su `float16`. Su CPU la precisione mista è disabilitata. Il `GradScaler` è attivo solo con `float16` su CUDA — `bfloat16` non richiede loss scaling.

#### Gradient Accumulation

Il gradiente viene accumulato per `accumulation_steps = 32` batch prima di ogni step dell'ottimizzatore. Con `batch_size = 8` (definito in `dataengine/configuration.py`, file separato e indipendente), il batch effettivo è **8 × 32 = 256 campioni**.

#### Gradient Clipping

La norma del gradiente è clippata a `max_norm = 1.0`. Se la `grad_norm` risulta `NaN` o `Inf`, l'intero blocco di aggiornamento viene saltato: nessun passo dell'ottimizzatore, nessun aggiornamento dello scheduler, garantendo la stabilità del training.

#### Learning Rate Scheduler — Cosine Decay con Warmup

Attivo quando `use_scheduler = True`. Il numero di step di warmup viene calcolato internamente come `warmup_steps = int(total_optimization_steps × warmup_ratio)`, dove `warmup_ratio` è configurabile in `trn_configuration`. Durante il warmup il LR cresce linearmente da 0 fino al valore iniziale; successivamente decade con cosine annealing fino a `lr_min_ratio × learning_rate`.

Se `use_scheduler = False`, lo scheduler applica una funzione costante λ = 1.0 e il **LR rimane fisso** per tutta la durata del training.

#### Salvataggio Progressivo

Il checkpoint viene salvato circa ogni 20% delle iterazioni di un'epoca (`save_step = max(1, total_batches // 5)`) e obbligatoriamente al termine di ogni epoca. Il modello con il miglior validation loss viene salvato separatamente in `output/bckpt.pth`.

#### Resume del Training

Al ripristino da checkpoint, i batch già processati nell'epoca corrente vengono saltati usando `itertools.islice` sull'iteratore del DataLoader. Questo evita di caricare e trasferire su GPU dati inutili, rendendo il resume efficiente anche su dataset di grandi dimensioni.

---

### 4.3 Validazione

**File:** `training/train.py` — `validate_model()`

La loss di validazione viene calcolata al termine di ogni epoca con il modello in modalità `eval()` e `torch.no_grad()`. La loss è pesata per il numero di token (non di batch) per un confronto corretto tra batch di dimensioni diverse. In presenza di loss `NaN` o `Inf` il metodo restituisce `float('inf')`.

---

## 5. Gestione degli State e dei Checkpoint

**File:** `utils/state_manager.py`, `utils/model_manager.py`

`save_state` salva un checkpoint completo che permette la ripresa esatta del training da qualsiasi punto:

| Campo | Descrizione |
|---|---|
| `model_state_dict` | Pesi del modello |
| `optimizer_state_dict` | Stato di AdamW (medie dei momenti) |
| `scheduler_state_dict` | Stato del LR scheduler |
| `scaler_state_dict` | Stato del GradScaler (AMP) |
| `epoch`, `batch` | Posizione corrente nel dataset |
| `current_opt_step` | Step di ottimizzazione corrente |
| `rng_state`, `cuda_rng_state` | Stato dei generatori random (CPU e GPU) |

**Salvataggio atomico:** il file viene scritto prima come `.tmp` e poi rinominato con `os.replace`, evitando checkpoint corrotti in caso di interruzione improvvisa.

**Nota:** `load_model` (in `model_manager.py`) è una versione leggera che carica solo i pesi del modello, pensata per l'inferenza. Se il file non esiste, la funzione restituisce silenziosamente senza sollevare eccezioni.

---

## 6. Entry Point — main.py

**File:** `main.py`

Orchestratore principale del progetto. Il flusso di esecuzione è:

```
1. Inizializzazione device (CUDA > CPU)
2. Creazione modello
3. Tokenizzazione del dataset (solo se i file .bin non esistono già)
4. Creazione DataLoader per training e validazione
5. Training (train_model)
6. Salvataggio del modello finale — solo pesi — in output/model.pth
```

---

## 7. Struttura del Progetto

```
project/
│
├── src/
│   ├── architecture/
│   │   ├── configuration.py           # Iperparametri del modello
│   │   ├── model.py                   # Classe Model principale
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
│   ├── parq/                          # Dataset raw in formato Parquet
│   └── bin/                           # Dataset tokenizzato in formato binario
│
├── output/
│   ├── lckpt.pth                      # Ultimo checkpoint
│   ├── bckpt.pth                      # Miglior checkpoint (val loss)
│   ├── model.pth                      # Modello finale (solo pesi)
│   └── training_results.json          # Metriche finali
│
└── docs/                              # Documentazione
```

---

## 8. Requisiti

| Pacchetto | Versione minima | Utilizzo |
|---|---|---|
| `torch` | >= 2.0 | Framework principale, AMP, SDPA / Flash Attention |
| `numpy` | qualsiasi | Lettura memmap e tokenizzazione binaria |
| `pandas` | qualsiasi | Lettura file Parquet |
| `tiktoken` | qualsiasi | Tokenizzatore GPT-2 BPE |
| `tqdm` | qualsiasi | Barre di avanzamento training |
| `pyarrow` | qualsiasi | Backend engine per la lettura Parquet |

> **Nota GPU:** per il training con mixed precision `bfloat16` è raccomandata una GPU con architettura NVIDIA Ampere o superiore (es. RTX 3000, A100). Su hardware precedente il training ricade automaticamente su `float16` con GradScaler.