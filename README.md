# Clinical Risk Prediction & Explainability Platform

Evolves a basic heart-disease classifier into a reliable, interpretable,
reproducible ML system: benchmarking → calibration → SHAP explainability →
reproducible pipeline → FastAPI service → fairness audit → (optional RAG layer).

## Phase plan

- [ ] Phase 1 — Data + baseline models (LR, RF, XGBoost)
- [ ] Phase 2 — Calibration + threshold optimization
- [ ] Phase 3 — Explainability (SHAP global + local)
- [ ] Phase 4 — Reproducibility + experiment tracking (MLflow)
- [ ] Phase 5 — FastAPI inference service + PostgreSQL history
- [ ] Phase 6 — Docker Compose + dashboard
- [ ] Phase 7 — Fairness audit across subgroups
- [ ] Phase 8 — RAG clinical context layer (optional, cut first if needed)

## Structure

```
data/
  raw/            # original dataset, untouched
  processed/      # train/val/test splits, cleaned data
src/
  preprocessing/  # cleaning, encoding, scaling
  training/       # model training scripts
  inference/      # prediction + FastAPI service (Phase 5+)
  evaluation/      # metrics, calibration, threshold analysis, SHAP
configs/          # YAML configs for reproducible runs
tests/
notebooks/        # exploratory work only — nothing here feeds the pipeline
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset

UCI/Cleveland Heart Disease dataset (1,025-record version, Kaggle mirror).
Place raw CSV at `data/raw/heart.csv` before running Phase 1 scripts.

## Running the full stack locally (Docker Compose)

```bash
docker compose up --build
```

- Frontend: http://localhost:5173
- API: http://localhost:8001 (docs at /docs)
- Postgres: localhost:5434 (user: clinical, db: clinical_risk)

## Running without Docker (dev mode)

Backend:
```bash
python -m src.inference.train_production_model   # only needed once, or after retraining
uvicorn src.inference.main:app --reload --port 8001
```

Frontend:
```bash
cd frontend
npm install
echo "VITE_API_URL=http://localhost:8001" > .env
npm run dev
```
