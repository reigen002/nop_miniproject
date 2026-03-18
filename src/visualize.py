"""
Visualization module for generating publication-quality plots.

Generates convergence curves, sparsity evolution, feature importance
bar charts, and MSE comparison plots for the technical report.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json


# Set publication-quality style
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

COLORS = {
    'Ridge': '#3498db',
    'LASSO': '#e74c3c',
    'APG-DST': '#2ecc71',
}


def plot_convergence(history, save_dir='results/plots'):
    """Plot objective value and MSE convergence over iterations."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Objective convergence
    ax = axes[0]
    iters = range(len(history['objective']))
    ax.semilogy(iters, history['objective'], color=COLORS['APG-DST'], linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective Value (log scale)')
    ax.set_title('Convergence of Composite Objective')
    ax.grid(True, alpha=0.3)

    # MSE convergence
    ax = axes[1]
    ax.plot(iters, history['mse'], color=COLORS['APG-DST'], linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('MSE During Training')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'convergence.png'))
    plt.close()
    print(f"[Plot] Saved convergence.png")


def plot_sparsity_evolution(history, n_features, save_dir='results/plots'):
    """Plot how sparsity and number of active features evolve."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    iters = range(len(history['sparsity']))

    # Sparsity percentage
    ax = axes[0]
    ax.plot(iters, history['sparsity'], color=COLORS['APG-DST'], linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Sparsity (%)')
    ax.set_title('Feature Sparsity Evolution')
    ax.set_ylim(-5, 105)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Number of non-zero features
    ax = axes[1]
    ax.plot(iters, history['n_nonzero'], color='#9b59b6', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Number of Active Features')
    ax.set_title('Active Feature Count')
    ax.axhline(y=n_features, color='gray', linestyle='--', alpha=0.5,
               label=f'Total features ({n_features})')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sparsity_evolution.png'))
    plt.close()
    print(f"[Plot] Saved sparsity_evolution.png")


def plot_lambda_adaptation(history, save_dir='results/plots'):
    """Plot how the adaptive regularization strength evolves."""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    iters = range(len(history['lambda_mean']))
    ax.plot(iters, history['lambda_mean'], color=COLORS['APG-DST'],
            linewidth=2, label='Mean λ')
    ax.fill_between(iters,
                    [0] * len(iters),
                    history['lambda_max'],
                    alpha=0.15, color=COLORS['APG-DST'], label='Max λ range')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Regularization Strength (λ)')
    ax.set_title('Dynamic Soft-Thresholding: Adaptive λ Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'lambda_adaptation.png'))
    plt.close()
    print(f"[Plot] Saved lambda_adaptation.png")


def plot_coefficient_comparison(all_results, feature_names, save_dir='results/plots'):
    """Compare coefficient magnitudes across all models."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(len(all_results), 1,
                              figsize=(14, 4 * len(all_results)),
                              sharex=True)

    if len(all_results) == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, all_results.items()):
        coef = res['coefficients']
        color = COLORS.get(name, '#95a5a6')

        ax.bar(range(len(coef)), np.abs(coef), color=color, alpha=0.7, width=1.0)
        ax.set_ylabel('|Coefficient|')
        ax.set_title(f'{name} — {np.sum(np.abs(coef) > 1e-8)} nonzero features')
        ax.grid(True, alpha=0.3, axis='y')

    axes[-1].set_xlabel('Feature Index')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'coefficient_comparison.png'))
    plt.close()
    print(f"[Plot] Saved coefficient_comparison.png")


def plot_mse_comparison(all_results, save_dir='results/plots'):
    """Bar chart comparing MSE across models."""
    os.makedirs(save_dir, exist_ok=True)

    models = list(all_results.keys())
    train_mse = [all_results[m]['train_metrics']['mse'] for m in models]
    val_mse = [all_results[m]['val_metrics']['mse'] for m in models]
    test_mse = [all_results[m]['test_metrics']['mse'] for m in models]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, train_mse, width, label='Train', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, val_mse, width, label='Validation', color='#f39c12', alpha=0.8)
    bars3 = ax.bar(x + width, test_mse, width, label='Test', color='#e74c3c', alpha=0.8)

    ax.set_xlabel('Model')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('MSE Comparison Across Models')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mse_comparison.png'))
    plt.close()
    print(f"[Plot] Saved mse_comparison.png")


def plot_sparsity_vs_mse(all_results, save_dir='results/plots'):
    """Scatter plot: sparsity vs MSE trade-off."""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, res in all_results.items():
        sparsity = res['test_metrics'].get('sparsity_pct', 0)
        mse = res['test_metrics']['mse']
        color = COLORS.get(name, '#95a5a6')

        ax.scatter(sparsity, mse, s=200, color=color, label=name,
                  edgecolors='black', linewidth=1.5, zorder=5)

    ax.set_xlabel('Sparsity (% zero coefficients)')
    ax.set_ylabel('Test MSE')
    ax.set_title('Sparsity vs. Predictive Accuracy Trade-off')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sparsity_vs_mse.png'))
    plt.close()
    print(f"[Plot] Saved sparsity_vs_mse.png")


def plot_feature_importance(all_results, feature_names, top_n=15,
                            save_dir='results/plots'):
    """Horizontal bar chart of top features by absolute coefficient."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, len(all_results),
                              figsize=(6 * len(all_results), 8))

    if len(all_results) == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, all_results.items()):
        coef = res['coefficients']
        abs_coef = np.abs(coef)
        top_idx = np.argsort(abs_coef)[-top_n:]
        top_names = [feature_names[i] for i in top_idx]
        top_vals = abs_coef[top_idx]
        color = COLORS.get(name, '#95a5a6')

        ax.barh(range(top_n), top_vals, color=color, alpha=0.8)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_names, fontsize=9)
        ax.set_xlabel('|Coefficient|')
        ax.set_title(f'Top {top_n} Features — {name}')
        ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_importance.png'))
    plt.close()
    print(f"[Plot] Saved feature_importance.png")


def plot_residual_distributions(all_results, y_test, save_dir='results/plots'):
    """Histogram of residual distributions for each model."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, len(all_results),
                              figsize=(6 * len(all_results), 5))

    if len(all_results) == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, all_results.items()):
        residuals = y_test - res['y_pred_test']
        color = COLORS.get(name, '#95a5a6')

        ax.hist(residuals, bins=40, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Residual (Actual - Predicted)')
        ax.set_ylabel('Count')
        ax.set_title(f'{name} Residuals\nμ={np.mean(residuals):.3f}, σ={np.std(residuals):.3f}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'residual_distributions.png'))
    plt.close()
    print(f"[Plot] Saved residual_distributions.png")


def plot_actual_vs_predicted(all_results, y_test, save_dir='results/plots'):
    """Scatter plot of actual vs predicted values for each model."""
    os.makedirs(save_dir, exist_ok=True)

    n_models = len(all_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, all_results.items()):
        y_pred = res['y_pred_test']
        color = COLORS.get(name, '#95a5a6')
        r2 = res['test_metrics']['r2']

        ax.scatter(y_test, y_pred, alpha=0.3, s=10, color=color)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, 'k--', linewidth=1.5, label='Perfect prediction')
        ax.set_xlabel('Actual Value')
        ax.set_ylabel('Predicted Value')
        ax.set_title(f'{name} — Actual vs Predicted\n(R² = {r2:.4f})')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'actual_vs_predicted.png'))
    plt.close()
    print(f"[Plot] Saved actual_vs_predicted.png")


def plot_convergence_with_phases(history, n_outer, warmup_frac=0.3, save_dir='results/plots'):
    """Convergence plot with warm-up (Phase 1) and adaptive (Phase 2) shading."""
    os.makedirs(save_dir, exist_ok=True)

    iters = list(range(len(history['objective'])))
    warmup_end = int(n_outer * warmup_frac)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, key, label in [
        (axes[0], 'objective', 'Composite Objective (log scale)'),
        (axes[1], 'mse',       'Mean Squared Error'),
    ]:
        vals = history[key]
        ax.axvspan(0, min(warmup_end, len(iters) - 1), alpha=0.12,
                   color='#3498db', label='Phase 1: Fixed λ (warm-up)')
        ax.axvspan(min(warmup_end, len(iters) - 1), len(iters) - 1, alpha=0.12,
                   color='#2ecc71', label='Phase 2: Adaptive λ')
        if key == 'objective':
            ax.semilogy(iters, vals, color=COLORS['APG-DST'], linewidth=2)
        else:
            ax.plot(iters, vals, color=COLORS['APG-DST'], linewidth=2)
        ax.axvline(x=warmup_end, color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(label)
        ax.set_title(f'APG-DST Convergence: {label}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'convergence_phases.png'))
    plt.close()
    print(f"[Plot] Saved convergence_phases.png")


def plot_gamma_sweep(sweep_results, save_dir='results/plots'):
    """Dual-axis plot: MSE and Sparsity vs gamma for sensitivity analysis."""
    if not sweep_results:
        return
    os.makedirs(save_dir, exist_ok=True)

    gammas     = [r['gamma']        for r in sweep_results]
    mses       = [r['test_mse']     for r in sweep_results]
    sparsities = [r['sparsity_pct'] for r in sweep_results]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    color_mse = COLORS['APG-DST']
    color_sp  = '#e74c3c'

    l1 = ax1.plot(gammas, mses, 'o-', color=color_mse, linewidth=2,
                  markersize=8, label='Test MSE')
    ax1.set_xlabel('Adaptive Exponent γ', fontsize=13)
    ax1.set_ylabel('Test MSE', color=color_mse, fontsize=13)
    ax1.tick_params(axis='y', labelcolor=color_mse)

    ax2 = ax1.twinx()
    l2 = ax2.plot(gammas, sparsities, 's--', color=color_sp, linewidth=2,
                  markersize=8, label='Sparsity (%)')
    ax2.set_ylabel('Sparsity (%)', color=color_sp, fontsize=13)
    ax2.tick_params(axis='y', labelcolor=color_sp)

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=10)
    ax1.set_title('APG-DST Hyperparameter Sensitivity: γ vs MSE & Sparsity', fontsize=14)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gamma_sweep.png'))
    plt.close()
    print(f"[Plot] Saved gamma_sweep.png")


def generate_all_plots(all_results, data, save_dir='results/plots',
                       gamma_sweep_results=None):
    """Generate all visualization plots."""
    os.makedirs(save_dir, exist_ok=True)
    feature_names = data['feature_names']
    y_test = data['y_test']
    n_features = data['n_total_features']

    print("\n" + "=" * 70)
    print(f"{'GENERATING VISUALIZATIONS':^70}")
    print("=" * 70)

    # APG-specific plots
    if 'APG-DST' in all_results and 'history' in all_results['APG-DST']:
        history = all_results['APG-DST']['history']
        n_outer = len(history['objective'])
        plot_convergence(history, save_dir)
        plot_convergence_with_phases(history, n_outer, save_dir=save_dir)
        plot_sparsity_evolution(history, n_features, save_dir)
        plot_lambda_adaptation(history, save_dir)

    # Comparison plots
    plot_coefficient_comparison(all_results, feature_names, save_dir)
    plot_mse_comparison(all_results, save_dir)
    plot_sparsity_vs_mse(all_results, save_dir)
    plot_feature_importance(all_results, feature_names, save_dir=save_dir)
    plot_residual_distributions(all_results, y_test, save_dir)
    plot_actual_vs_predicted(all_results, y_test, save_dir)

    # Gamma sensitivity (if sweep was run)
    if gamma_sweep_results:
        plot_gamma_sweep(gamma_sweep_results, save_dir)
    else:
        # Try to load from file
        sweep_file = os.path.join(os.path.dirname(save_dir), 'gamma_sweep.json')
        if os.path.exists(sweep_file):
            with open(sweep_file) as f:
                plot_gamma_sweep(json.load(f), save_dir)

    print(f"\n[Plots] All visualizations saved to '{save_dir}/'")
