"""
Unified training pipeline for all models.

Runs Ridge, LASSO, ElasticNet, and APG-DST on the same dataset and collects
results for comparison. Also includes a gamma sensitivity sweep for APG-DST.
"""

import json
import os
import warnings
import numpy as np
warnings.filterwarnings('ignore')
from src.data.dataset import load_house_prices
from src.models.baselines import train_ridge, train_lasso, train_elastic_net
from src.optimizers.adaptive_proximal import train_apg


def run_all_experiments(poly_degree=2, lambda_0=0.02, gamma=0.5,
                        apg_max_iter=30, results_dir='results'):

    """
    Run all three models on the House Prices dataset and save results.

    Parameters
    ----------
    poly_degree : int — polynomial expansion degree
    lambda_0 : float — APG base regularization
    gamma : float — APG adaptive weight exponent
    apg_max_iter : int — max iterations for APG
    results_dir : str — directory to save results

    Returns
    -------
    dict — all results keyed by model name
    """
    os.makedirs(results_dir, exist_ok=True)

    # --- Load dataset ---
    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)
    data = load_house_prices(poly_degree=poly_degree)

    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    feature_names = data['feature_names']

    all_results = {}

    # --- Ridge Regression ---
    print("\n" + "=" * 70)
    print("RIDGE REGRESSION")
    print("=" * 70)
    ridge_results = train_ridge(X_train, y_train, X_val, y_val, X_test, y_test)
    all_results['Ridge'] = ridge_results

    # --- Standard LASSO ---
    print("\n" + "=" * 70)
    print("STANDARD LASSO")
    print("=" * 70)
    lasso_results = train_lasso(X_train, y_train, X_val, y_val, X_test, y_test, max_iter=50000)
    all_results['LASSO'] = lasso_results

    # --- Elastic Net ---
    print("\n" + "=" * 70)
    print("ELASTIC NET (L1 + L2 Baseline)")
    print("=" * 70)
    en_results = train_elastic_net(X_train, y_train, X_val, y_val, X_test, y_test)
    all_results['ElasticNet'] = en_results

    # --- APG with Dynamic Soft-Thresholding + FISTA ---
    print("\n" + "=" * 70)
    print("ADAPTIVE PROXIMAL GRADIENT (APG-DST + FISTA)")
    print("=" * 70)
    apg_results = train_apg(
        X_train, y_train, X_val, y_val, X_test, y_test,
        lambda_0=lambda_0, gamma=gamma, max_iter=apg_max_iter
    )
    all_results['APG-DST'] = apg_results

    # --- Save comparison summary ---
    summary = _build_comparison_summary(all_results, feature_names)
    _save_results(summary, all_results, results_dir)
    _print_comparison_table(summary)

    return all_results, data


def run_gamma_sweep(poly_degree=2, lambda_0=0.02, gamma_values=None,
                   apg_max_iter=30, results_dir='results'):
    """
    Hyperparameter sensitivity: Run APG-DST with different gamma values.

    Tests how the adaptive weight exponent gamma affects sparsity and MSE.
    Results saved to results/gamma_sweep.json for the report.
    """
    if gamma_values is None:
        gamma_values = [0.25, 0.5, 0.75, 1.0, 1.5]

    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("GAMMA SENSITIVITY SWEEP (APG-DST)")
    print("=" * 70)
    print(f"  Testing gamma values: {gamma_values}")

    data = load_house_prices(poly_degree=poly_degree)
    sweep_results = []

    for gamma in gamma_values:
        print(f"\n  --- gamma = {gamma} ---")
        res = train_apg(
            data['X_train'], data['y_train'],
            data['X_val'],   data['y_val'],
            data['X_test'],  data['y_test'],
            lambda_0=lambda_0, gamma=gamma, max_iter=apg_max_iter,
            verbose=False
        )
        tm = res['test_metrics']
        entry = {
            'gamma':        gamma,
            'test_mse':     tm['mse'],
            'test_r2':      tm['r2'],
            'sparsity_pct': tm.get('sparsity_pct', 0),
            'n_nonzero':    tm.get('n_nonzero', 0),
            'train_time':   res['train_time'],
        }
        sweep_results.append(entry)
        print(f"    MSE={entry['test_mse']:.4f}, Sparsity={entry['sparsity_pct']:.1f}%, "
              f"Nonzero={entry['n_nonzero']}, Time={entry['train_time']:.1f}s")

    with open(os.path.join(results_dir, 'gamma_sweep.json'), 'w') as f:
        import json
        json.dump(sweep_results, f, indent=2)
    print(f"\n[GammaSweep] Saved to '{results_dir}/gamma_sweep.json'")
    return sweep_results


def _build_comparison_summary(all_results, feature_names):
    """Build a summary comparison dict."""
    summary = {}
    for name, res in all_results.items():
        tm = res['test_metrics']
        summary[name] = {
            'test_mse': tm['mse'],
            'test_rmse': tm['rmse'],
            'test_mae': tm['mae'],
            'test_r2': tm['r2'],
            'sparsity_pct': tm.get('sparsity_pct', 0),
            'n_nonzero': tm.get('n_nonzero', tm['n_features']),
            'n_features': tm.get('n_features', 0),
            'train_time': res['train_time'],
        }
    return summary


def _print_comparison_table(summary):
    """Print a formatted comparison table."""
    print("\n" + "=" * 90)
    print(f"{'MODEL COMPARISON':^90}")
    print("=" * 90)
    print(f"{'Model':<12} | {'Test MSE':>10} | {'Test RMSE':>10} | {'Test R²':>10} | "
          f"{'Sparsity':>10} | {'Nonzero':>8} | {'Time(s)':>8}")
    print("-" * 90)

    for name, s in summary.items():
        print(f"{name:<12} | {s['test_mse']:>10.4f} | {s['test_rmse']:>10.4f} | "
              f"{s['test_r2']:>10.4f} | {s['sparsity_pct']:>9.1f}% | "
              f"{s['n_nonzero']:>8d} | {s['train_time']:>8.3f}")

    print("=" * 90)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _save_results(summary, all_results, results_dir):
    """Save results to JSON."""
    # Save summary
    with open(os.path.join(results_dir, 'comparison_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    # Save APG convergence history
    if 'APG-DST' in all_results and 'history' in all_results['APG-DST']:
        history = {}
        for key, val in all_results['APG-DST']['history'].items():
            history[key] = [float(v) if not (isinstance(v, float) and (v != v)) else 0.0 for v in val]
        with open(os.path.join(results_dir, 'apg_history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    # Save coefficients via BytesIO to avoid numpy auto-appending .npy
    # and macOS file-locking issues on the results/ directory.
    import io
    for name, res in all_results.items():
        coef = res['coefficients']
        fname = f'{name.lower().replace("-","_")}_coefficients.npy'
        final_path = os.path.join(results_dir, fname)
        buf = io.BytesIO()
        np.save(buf, coef)
        raw = buf.getvalue()
        try:
            with open(final_path, 'wb') as fout:
                fout.write(raw)
        except PermissionError:
            alt = os.path.join(results_dir, fname.replace('.npy', '_v2.npy'))
            with open(alt, 'wb') as fout:
                fout.write(raw)

    print(f"\n[Results] Saved to '{results_dir}/' directory")
