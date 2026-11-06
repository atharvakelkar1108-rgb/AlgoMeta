# Intelligent ML Algorithm Advisor
### Meta-Learning Powered Automated Machine Learning Recommendation System

**End-Semester Project**

## Overview
Meta-learning system that recommends the best ML classification algorithms for a given tabular dataset by extracting 23 meta-features and using a trained meta-model (R² ≈ 0.95).

## New Features / Novelties Added
1. Multi-Objective Ranking (Performance + Speed + Interpretability)
2. Dataset Similarity Search
3. One-Click Code Generation (ready-to-run sklearn pipeline)
4. Meta-Model Explainability (feature importances)
5. Continual Learning support
6. Proper packaging (requirements, README, trained model)

## Setup & Run
```bash
cd ml_project
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Optional: retrain meta-model
```bash
python src/train_meta_model.py
```
