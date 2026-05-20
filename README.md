# Explainable Hybrid Movie Recommender with Knowledge Graph Reasoning and Transformer Attention

A 3-branch deep recommender system combining collaborative filtering, frozen DistilBERT plot embeddings, and TransE knowledge graph embeddings, with LIME-based explanations for every prediction.

University of Milano-Bicocca — *Advanced AI, Machine Learning and Deep Learning* (2026)

## Datasets
- **MovieLens 1M** — 1,000,209 ratings, 6,040 users, 3,883 movies
- **The Movies Dataset (Kaggle)** — plot summaries joined via TMDB IDs

After joining: 3,657 movies with plot text, 997,313 ratings retained.

## Architecture
- **Branch 1 — Collaborative filtering:** Neural Collaborative Filtering with learned user/movie embeddings.
- **Branch 2 — Plot text:** Frozen pretrained DistilBERT → 768-d [CLS] embeddings → projected to 64-d.
- **Branch 3 — Knowledge graph:** TransE on a 10,138-triple movie–genre–decade KG → 50-d entity embeddings.
- **Fusion:** concatenate → MLP → predicted rating.
- **XAI:** LIME on plot text + branch-level ablation + counterfactual top-K recommendations.

## Results (Test RMSE)

| Model              | Test RMSE | Δ vs baseline |
|--------------------|-----------|---------------|
| NCF only           | 0.9171    | —             |
| NCF + KG           | 0.9164    | +0.0007       |
| **NCF + Text**     | **0.9004** | **+0.0167** |
| NCF + Text + KG    | 0.9028    | +0.0143       |

The text branch produced a 1.7% RMSE improvement. Combining text + KG slightly underperformed text alone — a feature-redundancy finding (DistilBERT plot embeddings implicitly encode genre and era information).

## How to run
1. Open `Movie_Recommender_Project.ipynb` in Google Colab.
2. Set runtime to GPU (T4 is enough).
3. Run cells top to bottom. The notebook auto-downloads MovieLens 1M and prompts once for a Kaggle API token to fetch The Movies Dataset.

## Course topics covered
Recommender Systems · Representation Learning · Transfer Learning · Attention / Transformers · Knowledge Graphs and Reasoning · Explainable AI

## Stack
PyTorch · HuggingFace Transformers · LIME · pandas · scikit-learn
