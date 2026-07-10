# Explainable Hybrid Movie Recommender with Knowledge Graph Reasoning and Transformer Attention

A movie rating predictor that fuses **three signals** — collaborative filtering, transformer-based plot understanding (DistilBERT), and a knowledge graph (TransE) — and **explains every prediction** with three Explainable-AI methods.

> Course project — *Advanced Artificial Intelligence, Machine Learning and Deep Learning* (F9103Q009), University of Milano-Bicocca, 2025–2026.

---

## Overview

Standard collaborative-filtering recommenders have two weaknesses: they ignore *what a movie is about*, and they are black boxes that cannot explain their recommendations. This project addresses both by adding two content-aware branches (plot text and a knowledge graph) on top of a neural collaborative-filtering baseline, and by making every prediction interpretable.

The project demonstrates six course topics in one system: **Recommender Systems, Representation Learning, Transfer Learning, Attention/Transformers, Knowledge Graphs & Reasoning, and Explainable AI.**

---

## Datasets

Two public datasets, joined via the `links.csv` bridge file:

| Dataset | Type | Provides |
|---|---|---|
| **MovieLens 1M** | Behavioral | user–movie–rating triples (1M ratings) |
| **The Movies Dataset (Kaggle)** | Content | plot summaries, genres (45K movies) |

**Join:** `MovieLens movie_id → links.movieId → links.tmdbId → movies_metadata.id`
**After joining:** 6,040 users · 3,657 movies · 997,313 ratings (99.7% retained).

---

## Architecture

```
   Branch 1  User + Movie embeddings (collaborative)          64 + 64
   Branch 2  Plot text -> frozen DistilBERT [CLS] (768) -> standardize -> 64
   Branch 3  Knowledge graph -> TransE embedding (50) -> 64
                              |
                     concatenate (fusion)
                              |
                     MLP (128 -> 64 -> 1)
                              |
                     predicted rating (1-5)
                              |
        Explainable AI: LIME + branch ablation + top-K vs ground truth
```

All four neural variants share the **same fusion head and training protocol** — only the inputs change — so the comparison is a fair ablation.

---

## Results (test set, full data)

| Model | RMSE | MAE | P@10 | R@10 |
|---|---:|---:|---:|---:|
| Global Mean | 1.1139 | 0.9312 | 0.0213 | 0.0237 |
| Popularity | 0.9781 | 0.7819 | 0.0000 | 0.0000 |
| Memory User-CF | 0.8911 | 0.6972 | 0.0044 | 0.0021 |
| NCF | 0.8886 | 0.6978 | 0.0372 | 0.0474 |
| NCF + KG | 0.8861 | 0.6991 | 0.0356 | 0.0420 |
| **NCF + Text** | **0.8660** | **0.6787** | **0.0463** | **0.0565** |
| NCF + Text + KG | 0.8818 | 0.6962 | 0.0370 | 0.0451 |

**Best model: NCF + Text (RMSE 0.8660)** — wins on all four metrics.

---

## Key finding

**More signals are not always better — what matters is whether they add new information.**

- Plot text gave the largest single improvement (NCF 0.8886 → NCF+Text **0.8660**).
- The knowledge graph alone barely helped (0.8886 → 0.8861), and adding it *on top of* the text branch made things **worse** (0.8660 → 0.8818).
- **Why?** Feature redundancy: DistilBERT plot embeddings already encode genre. We confirmed this with an **NMI test** — clustering movies using only the plot embeddings recovers the genres (NMI = 0.13, clearly above random), so the explicit knowledge-graph genre triples are redundant.

We also diagnosed and fixed an **overfitting** issue: raw DistilBERT `[CLS]` vectors are large and anisotropic and caused the text branch to overfit (validation RMSE diverged). Standardizing the embeddings (mean 0, std 1) fixed it and made the text branch the best model.

---

## Explainable AI

Three complementary methods explain every prediction at different levels:

1. **LIME** — which plot *words* drove the rating (re-encodes perturbed plots through the same frozen DistilBERT).
2. **Branch ablation** — which *branch* mattered (zeroing each branch; the KG changes predictions least, confirming redundancy).
3. **Top-K vs ground truth** — the model's top-10 recommendations compared with what the user actually rated highly (the human-readable form of Precision@10 / Recall@10).

---

## How to run

1. Open `Movie_Recommender_Project.ipynb` in Google Colab.
2. Runtime → set a **T4 GPU**.
3. Run all cells top to bottom.

The notebook is **dual-mode**:
- If a local `Data/` folder with the sample CSVs is present, it runs in ~5 minutes (verification).
- Otherwise it downloads the full datasets (MovieLens is public; The Movies Dataset needs a Kaggle token entered once) and reproduces the full results in ~15 minutes.

A single global seed (`SEED = 42`) makes the results reproducible; embeddings are cached so re-runs are fast.

---

## Repository contents

| File | Description |
|---|---|
| `Movie_Recommender_Project.ipynb` | Full project notebook (report + code + outputs) |
| `Project_Report.pdf` | Written report with figures and results |
| `Data/` | Small representative data sample for quick verification |
| `README.md` | This file |

---

## Team & contributions

- **Subir Saha** — data pipeline, baselines (Global Mean, Popularity, Memory User-CF), Knowledge Graph + TransE branch, reproducibility.
- **Md Rifatul Haque** — plot-text (DistilBERT) branch, hybrid fusion model, evaluation pipeline (RMSE/MAE/P@10/R@10), Explainable AI, limitations.

## Tech stack

Python · PyTorch · HuggingFace Transformers · scikit-learn · LIME · pandas · matplotlib · Google Colab

## AI usage

Generative AI (Claude) was used as a coding and writing assistant. All experiments were run and verified by the team; all reported numbers come from the team's own executed runs. See the AI Usage Declaration in the notebook.
