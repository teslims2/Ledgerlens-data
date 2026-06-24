# scripts/

## `generate_synthetic_dataset.py`

Generates a synthetic labelled feature matrix for local training, demos,
and tests, without needing live Stellar Horizon data.

The output schema matches `detection/feature_engineering.py::build_feature_matrix`
(`wallet` + all Benford / trade-pattern / volume-timing / wallet-graph
feature columns), plus a `label` column (`1` = wash-trading-like, `0` =
legitimate). Roughly half the rows are generated with "legitimate"
distributions and half with "wash-trading-like" distributions, then
shuffled.

### Usage

```bash
python -m scripts.generate_synthetic_dataset \
    --n-wallets 500 \
    --seed 42 \
    --output data/synthetic_dataset.parquet
```

| Flag | Default | Description |
|---|---|---|
| `--n-wallets` | `500` | Number of synthetic wallet rows to generate |
| `--seed` | `42` | Random seed (controls both data generation and the final shuffle) |
| `--output` | `data/synthetic_dataset.parquet` | Output parquet path |

### Training on the generated dataset

```bash
python -m detection.model_training --data-path data/synthetic_dataset.parquet
```

This trains every model in `MODEL_REGISTRY` (Random Forest, XGBoost,
LightGBM) with SMOTE-balanced training data, writes the fitted models to
`config.MODEL_DIR`, and writes `metrics.json` (AUC-ROC / PR-AUC / F1 per
model) alongside them.

## `run_adversarial_eval.py`

Generates an adversarial-robustness report for a trained ensemble. It runs
FGSM and PGD evasion attacks (`detection/adversarial/`) against the
high-scoring wash wallets in a labelled feature matrix and writes a JSON
report covering:

- PGD / FGSM evasion success rate (fraction of `80+` wash wallets pushed
  below the alert threshold within the L-inf budget),
- per-feature minimum epsilon and the most vulnerable features, and
- the AUC-ROC gain from adversarial-augmentation retraining.

### Usage

```bash
python -m scripts.run_adversarial_eval \
    --data-path data/synthetic_dataset.parquet \
    --model-dir ./models \
    --output reports/adversarial_robustness.json
```

| Flag | Default | Description |
|---|---|---|
| `--data-path` | *(required)* | Labelled feature matrix (parquet) with a `label` column |
| `--model-dir` | `MODEL_DIR` | Directory of trained model artifacts |
| `--output` | `reports/adversarial_robustness.json` | Output JSON report path |
| `--epsilon` | `3.0` | L-inf perturbation budget (per-feature std units) |
| `--steps` | `40` | PGD iterations |
| `--target-score` | `40` | Evasion succeeds when the score drops below this |
| `--high-score` | `80` | Minimum score for a wallet to enter the attacked cohort |
| `--skip-augmentation` | off | Skip the slower adversarial-augmentation retraining comparison |

Requires trained models (run `model_training.py` first).
