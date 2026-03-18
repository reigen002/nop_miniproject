# NOP Mini Project - Theme 3

Dynamic Soft-Thresholding for Feature Selection in High-Dimensional Regression.

This project compares classical regularized linear models (Ridge, LASSO, Elastic Net) with a custom Adaptive Proximal Gradient method (APG-DST + FISTA) on a high-dimensional housing regression task.

## Project Goals

- Build a high-dimensional regression benchmark using polynomial feature expansion.
- Train and compare multiple sparse/regularized regression approaches.
- Analyze sparsity-accuracy trade-offs.
- Generate report-ready metrics and plots.
- Provide an interactive Flask dashboard for result exploration and price prediction.

## Tech Stack

- Python 3.10+
- NumPy, Pandas
- scikit-learn
- Matplotlib, Seaborn
- Flask

## Repository Structure

```text
NOP_Mini_Project_Code_Submission/
  main.py                     # End-to-end experiment runner
  requirements.txt            # Python dependencies
  src/
    train.py                  # Unified training pipeline and gamma sweep
    evaluate.py               # Detailed evaluation report generation
    visualize.py              # Plot generation utilities
    data/dataset.py           # Dataset loading + preprocessing
    models/baselines.py       # Ridge/LASSO/ElasticNet baselines
    optimizers/adaptive_proximal.py  # APG-DST optimizer
  dashboard/
    app.py                    # Flask dashboard app
    templates/                # Dashboard HTML templates
  results/                    # Generated outputs (JSON, NPY, PNG)
```

## Setup

1. Open a terminal in the project root.
2. (Optional) Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run The Full Pipeline

Run with defaults:

```bash
python main.py
```

Common options:

```bash
python main.py --poly_degree 2 --lambda_0 0.02 --gamma 0.5 --max_iter 30
```

Skip gamma sweep for a faster run:

```bash
python main.py --skip_gamma_sweep
```

### CLI Arguments

- `--poly_degree`: Polynomial expansion degree (default: `2`)
- `--lambda_0`: Base regularization strength for APG (default: `0.02`)
- `--gamma`: Adaptive weight exponent (default: `0.5`)
- `--max_iter`: APG outer iterations (default: `30`)
- `--results_dir`: Output folder (default: `results`)
- `--skip_gamma_sweep`: Disable APG gamma sensitivity sweep

## Launch The Dashboard

After generating results:

```bash
python dashboard/app.py
```

Open: http://localhost:8000

Dashboard pages include:

- Home summary
- APG convergence analysis
- Feature selection analysis
- Cross-model comparison
- Interactive prediction form

## Output Artifacts

The pipeline writes outputs under `results/`.

Core files:

- `comparison_summary.json`: compact metric comparison by model
- `evaluation_report.json`: detailed per-model report (metrics, residual stats, top features)
- `apg_history.json`: APG optimization history
- `ridge_coefficients.npy`, `lasso_coefficients.npy`, `elasticnet_coefficients.npy`, `apg_dst_coefficients.npy`
- `gamma_sweep.json` (if sweep is enabled)

Generated plots in `results/plots/`:

- `convergence.png`
- `convergence_phases.png`
- `sparsity_evolution.png`
- `lambda_adaptation.png`
- `coefficient_comparison.png`
- `mse_comparison.png`
- `sparsity_vs_mse.png`
- `feature_importance.png`
- `residual_distributions.png`
- `actual_vs_predicted.png`
- `gamma_sweep.png` (if sweep data is available)

## Notes

- The current data pipeline uses scikit-learn California Housing data and expands features polynomially to create a high-dimensional setting.
- The code is organized so the dataset loader can be swapped if required.

## Quick Start

```bash
pip install -r requirements.txt
python main.py
python dashboard/app.py
```
