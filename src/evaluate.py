"""
Evaluation module for final model assessment.

Generates detailed per-model evaluation reports including
predictions, residual analysis, and feature importance.
"""

import numpy as np
import json
import os
from src.models.baselines import compute_metrics


def evaluate_all_models(all_results, data, results_dir='results'):
    """
    Comprehensive evaluation of all models on test data.

    Parameters
    ----------
    all_results : dict — results from training pipeline
    data : dict — dataset with feature names
    results_dir : str — output directory

    Returns
    -------
    dict — evaluation report for each model
    """
    os.makedirs(results_dir, exist_ok=True)

    feature_names = data['feature_names']
    y_test = data['y_test']
    eval_report = {}

    print("\n" + "=" * 70)
    print(f"{'DETAILED EVALUATION REPORT':^70}")
    print("=" * 70)

    for name, res in all_results.items():
        metrics = res['test_metrics']
        coef = res['coefficients']

        # Feature importance analysis
        sorted_idx = np.argsort(np.abs(coef))[::-1]
        top_features = []
        for i, idx in enumerate(sorted_idx[:15]):
            if abs(coef[idx]) > 1e-8:
                top_features.append({
                    'rank': i + 1,
                    'name': feature_names[idx],
                    'coefficient': float(coef[idx]),
                    'abs_coefficient': float(abs(coef[idx])),
                })

        # Residual analysis
        y_pred = res['y_pred_test']
        residuals = y_test - y_pred
        residual_stats = {
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'median': float(np.median(residuals)),
            'max_abs': float(np.max(np.abs(residuals))),
            'q25': float(np.percentile(residuals, 25)),
            'q75': float(np.percentile(residuals, 75)),
        }

        model_report = {
            'name': name,
            'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else int(v)
                       for k, v in metrics.items()},
            'top_features': top_features,
            'residual_stats': residual_stats,
            'train_time': float(res['train_time']),
        }

        eval_report[name] = model_report

        # Print summary
        print(f"\n--- {name} ---")
        print(f"  MSE:  {metrics['mse']:.4f}   RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE:  {metrics['mae']:.4f}   R²:   {metrics['r2']:.4f}")
        print(f"  Sparsity: {metrics.get('sparsity_pct', 0):.1f}% "
              f"({metrics.get('n_nonzero', '?')}/{metrics.get('n_features', '?')} features)")
        print(f"  Residuals: mean={residual_stats['mean']:.4f}, std={residual_stats['std']:.4f}")
        if top_features:
            print(f"  Top 5 features:")
            for tf in top_features[:5]:
                print(f"    {tf['rank']}. {tf['name']}: {tf['coefficient']:.4f}")

    # Save full report
    with open(os.path.join(results_dir, 'evaluation_report.json'), 'w') as f:
        json.dump(eval_report, f, indent=2, default=str)

    print(f"\n[Evaluation] Full report saved to '{results_dir}/evaluation_report.json'")

    return eval_report
