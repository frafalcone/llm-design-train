# Custom LM — Risultati benchmark / Benchmark Results

> 🇮🇹 [Italiano](#italiano) · 🇬🇧 [English](#english)

---

<a name="italiano"></a>
## Italiano

Modello linguistico decoder-only addestrato da zero su una singola GPU consumer (AMD RX 9070 XT). L'architettura è un transformer in scala GPT-2 con scelte moderne: Grouped Query Attention (GQA) e RMSNorm al posto di MHA e LayerNorm standard.

---

### Architettura

| Parametro | Valore |
|---|---|
| Dimensione embedding | 768 |
| Layer | 12 |
| Teste di attenzione | 12 |
| Gruppi GQA | 4 |
| Lunghezza contesto | 1024 |
| Dimensione vocabolario | 50 257 |
| Espansione FFN | 4× |
| Normalizzazione | RMSNorm (ε=1e-6) |
| Bias | False |
| Dropout | 0.1 |
| Parametri stimati | ~85–90M |

Scelte chiave rispetto a GPT-2 vanilla: **Grouped Query Attention** (3 teste per gruppo, riduzione 3× della KV-cache), **RMSNorm** pre-norm (training più stabile), nessun termine di bias nell'intera rete.

---

### Risultati benchmark

#### Punteggi generali vs GPT-2

| Benchmark | Custom | GPT-2 | Delta |
|---|---|---|---|
| **BLiMP** | **74.03%** | 59.07% | **+14.96 pp** |
| **HellaSwag** | **28.09%** | 24.26% | **+3.83 pp** |
| **ARC-Easy** | **30.53%** | 28.77% | **+1.76 pp** |
| CoLA | 27.76% | 26.22% | +1.54 pp |
| LAMBADA | 13.62% | 17.40% | −3.78 pp |
| ARC-Challenge | 21.74% | 28.76% | −7.02 pp |

---

#### BLiMP — conoscenza grammaticale (67 sottocategorie)

Totale: **74.03%** vs GPT-2 59.07%

**Migliori sottocategorie**

| Sottocategoria | Punteggio |
|---|---|
| principle_A_case_1 | 100.00% |
| sentential_negation_npi_licensor_present | 99.80% |
| anaphor_number_agreement | 98.70% |
| determiner_noun_agreement_2 | 98.50% |
| irregular_past_participle_adjectives | 98.20% |
| wh_vs_that_no_gap_long_distance | 97.70% |
| principle_A_domain_1 | 97.10% |
| wh_vs_that_no_gap | 96.70% |
| anaphor_gender_agreement | 96.20% |
| determiner_noun_agreement_1 | 95.80% |

**Sottocategorie più deboli**

| Sottocategoria | Punteggio |
|---|---|
| wh_vs_that_with_gap_long_distance | 17.30% |
| only_npi_scope | 20.10% |
| principle_A_reconstruction | 28.60% |
| matrix_question_npi_licensor_present | 29.20% |
| existential_there_quantifiers_2 | 33.10% |
| distractor_agreement_relative_clause | 40.80% |
| sentential_subject_island | 43.00% |
| drop_argument | 45.80% |
| complex_NP_island | 46.60% |
| left_branch_island_echo_question | 50.30% |

Il modello mostra una forte competenza morfologica e di accordo (Principio A), con debolezze costanti sullo scope degli NPI e sulle dipendenze sintattiche a lunga distanza — tipiche di modelli piccoli addestrati su dati limitati.

---

#### CoLA — accettabilità linguistica (2 sottocategorie)

Totale: **27.76%**

| Sottocategoria | Custom | GPT-2 |
|---|---|---|
| In-domain | 55.51% | 52.44% |
| Out-of-domain | 0.00% | 0.00% |

Entrambi i modelli falliscono completamente fuori dominio. Le prestazioni in-domain sono modeste ma superiori a GPT-2.

---

#### LAMBADA — predizione ultima parola (1 sottocategoria)

Totale: **13.62%** (GPT-2: 17.40%)

Inferiore a GPT-2. LAMBADA penalizza i modelli privi di memoria contestuale a lungo raggio — un problema di quantità di dati più che di architettura.

---

#### HellaSwag — ragionamento di senso comune (192 sottocategorie)

Totale: **28.09%** vs GPT-2 24.26%

La distribuzione dei punteggi nelle 192 categorie è simile tra i due modelli, con entrambi concentrati nella fascia 20–40%. Nessun modello supera stabilmente il 50% in una singola categoria, risultato atteso a questa scala e con questo budget di dati.

---

#### ARC — domande scientifiche

| Benchmark | Custom | GPT-2 |
|---|---|---|
| ARC-Easy | 30.53% | 28.77% |
| ARC-Challenge | 21.74% | 28.76% |

Il modello vince su ARC-Easy ma perde su ARC-Challenge. Quest'ultimo richiede ragionamento fattuale a più passi che beneficia di corpora di pre-training più grandi.

---

### Analisi

**Punti di forza**
- Competenza grammaticale (BLiMP) nettamente superiore a GPT-2 a parità di scala
- Forte accordo morfologico e fenomeni di binding (determinante-sostantivo, anaforici, Principio A)
- Completamento di senso comune (HellaSwag) migliorato rispetto alla baseline

**Punti di debolezza**
- Dipendenze a lunga distanza e licensing degli NPI ancora difficili
- Recupero fattuale e ragionamento multi-step (ARC-Challenge, LAMBADA) limitati dal volume di dati
- Generalizzazione CoLA out-of-domain: zero — comune a questa scala

**Causa principale dei gap**: la maggior parte delle sotto-prestazioni è riconducibile al compute di training piuttosto che all'architettura. Il dataset contiene 1.38B token addestrati per 2 epoche (~2.76B token-update), ancora modesto rispetto agli standard moderni. Aumentare le epoche o la dimensione del dataset chiuderebbe probabilmente i gap su LAMBADA e ARC-Challenge senza modifiche architetturali.

---

### Setup di training

| | |
|---|---|
| Hardware | AMD RX 9070 XT (singola GPU) |
| Tipo di training | Pre-training da zero |
| Dataset — train | 1 378 513 680 token (~1.38B) |
| Dataset — valid | 74 230 744 token (~74.2M) |
| Epoche | 2 |
| Token-update effettivi | ~2.76B |
| Confrontato con | GPT-2 (117M, OpenAI 2019) |

---
---

<a name="english"></a>
## English

A decoder-only language model trained from scratch on a single consumer GPU (AMD RX 9070 XT). Architecture is a modernised GPT-2-scale transformer with Grouped Query Attention (GQA) and RMSNorm.

---

### Architecture

| Parameter | Value |
|---|---|
| Embedding dimension | 768 |
| Layers | 12 |
| Attention heads | 12 |
| GQA groups | 4 |
| Context length | 1024 |
| Vocabulary size | 50 257 |
| FFN expansion rate | 4× |
| Normalisation | RMSNorm (ε=1e-6) |
| Bias | False |
| Dropout | 0.1 |
| Estimated parameters | ~85–90M |

Key design choices over vanilla GPT-2: **Grouped Query Attention** (3 heads per group, 3× KV-cache reduction), **RMSNorm** pre-norm (more stable training), no bias terms throughout.

---

### Benchmark Results

#### Overall scores vs GPT-2

| Benchmark | Custom | GPT-2 | Delta |
|---|---|---|---|
| **BLiMP** | **74.03%** | 59.07% | **+14.96 pp** |
| **HellaSwag** | **28.09%** | 24.26% | **+3.83 pp** |
| **ARC-Easy** | **30.53%** | 28.77% | **+1.76 pp** |
| CoLA | 27.76% | 26.22% | +1.54 pp |
| LAMBADA | 13.62% | 17.40% | −3.78 pp |
| ARC-Challenge | 21.74% | 28.76% | −7.02 pp |

---

#### BLiMP — grammatical knowledge (67 subcategories)

Overall: **74.03%** vs GPT-2 59.07%

**Top subcategories**

| Subcategory | Score |
|---|---|
| principle_A_case_1 | 100.00% |
| sentential_negation_npi_licensor_present | 99.80% |
| anaphor_number_agreement | 98.70% |
| determiner_noun_agreement_2 | 98.50% |
| irregular_past_participle_adjectives | 98.20% |
| wh_vs_that_no_gap_long_distance | 97.70% |
| principle_A_domain_1 | 97.10% |
| wh_vs_that_no_gap | 96.70% |
| anaphor_gender_agreement | 96.20% |
| determiner_noun_agreement_1 | 95.80% |

**Weakest subcategories**

| Subcategory | Score |
|---|---|
| wh_vs_that_with_gap_long_distance | 17.30% |
| only_npi_scope | 20.10% |
| principle_A_reconstruction | 28.60% |
| matrix_question_npi_licensor_present | 29.20% |
| existential_there_quantifiers_2 | 33.10% |
| distractor_agreement_relative_clause | 40.80% |
| sentential_subject_island | 43.00% |
| drop_argument | 45.80% |
| complex_NP_island | 46.60% |
| left_branch_island_echo_question | 50.30% |

The model shows strong morphological agreement and binding (Principle A), with consistent weaknesses on NPI scope and long-distance syntactic dependencies — typical of small models trained on limited data.

---

#### CoLA — linguistic acceptability (2 subcategories)

Overall: **27.76%**

| Subcategory | Custom | GPT-2 |
|---|---|---|
| In-domain | 55.51% | 52.44% |
| Out-of-domain | 0.00% | 0.00% |

Both models fail entirely on out-of-domain. In-domain performance is modest but above GPT-2.

---

#### LAMBADA — last-word prediction (1 subcategory)

Overall: **13.62%** (GPT-2: 17.40%)

Below GPT-2. LAMBADA penalises models that lack long-range contextual memory — a data-quantity problem more than an architecture one.

---

#### HellaSwag — commonsense reasoning (192 subcategories)

Overall: **28.09%** vs GPT-2 24.26%

Score distribution across the 192 activity categories is broadly similar between the two models, with both clustering in the 20–40% range. Neither model reliably exceeds 50% in any single category, which is expected at this scale and data budget.

---

#### ARC — science Q&A

| Benchmark | Custom | GPT-2 |
|---|---|---|
| ARC-Easy | 30.53% | 28.77% |
| ARC-Challenge | 21.74% | 28.76% |

Custom wins on ARC-Easy but loses significantly on ARC-Challenge. ARC-Challenge requires multi-step factual reasoning that benefits from larger pretraining corpora.

---

### Analysis

**Strengths**
- Grammatical competence (BLiMP) well above GPT-2 despite the same parameter scale
- Strong morphological and agreement phenomena (determiner-noun, anaphor, Principle A)
- Commonsense completion (HellaSwag) improved over baseline

**Weaknesses**
- Long-range dependencies and NPI licensing remain difficult
- Factual recall and multi-step reasoning (ARC-Challenge, LAMBADA) limited by training data volume
- CoLA out-of-domain generalisation: zero — common at this scale

**Root cause of gaps**: Most underperformance traces to training compute rather than architecture. The dataset contains 1.38B tokens trained for 2 epochs (~2.76B token-updates), which is still modest by modern standards. Scaling epochs or dataset size would likely close the LAMBADA and ARC-Challenge gaps without any architectural changes.

---

### Training setup

| | |
|---|---|
| Hardware | AMD RX 9070 XT (single GPU) |
| Training type | Pre-training from scratch |
| Dataset — train | 1 378 513 680 token (~1.38B) |
| Dataset — valid | 74 230 744 token (~74.2M) |
| Epochs | 2 |
| Effective token-updates | ~2.76B |
| Compared against | GPT-2 (117M, OpenAI 2019) |
