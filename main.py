"""
Main entry point for the NOP Mini Project — Theme 3.

Dynamic Soft-Thresholding for Feature Selection in High-Dimensional Regression.

Usage:
    python main.py                      # Run with defaults
    python main.py --lambda_0 0.05      # Custom regularization
    python main.py --poly_degree 3      # Higher-dimensional features
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train import run_all_experiments, run_gamma_sweep
from src.evaluate import evaluate_all_models
from src.visualize import generate_all_plots


def main():
    parser = argparse.ArgumentParser(
        description='Theme 3: Dynamic Soft-Thresholding for Feature Selection'
    )
    parser.add_argument('--poly_degree', type=int, default=2,
                        help='Polynomial feature expansion degree (default: 2)')
    parser.add_argument('--lambda_0', type=float, default=0.02,
                        help='Base regularization strength for APG (default: 0.02)')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='Adaptive weight exponent (default: 0.5)')
    parser.add_argument('--max_iter', type=int, default=30,
                        help='Outer reweighting iterations for APG (default: 30)')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Output directory for results (default: results)')
    parser.add_argument('--skip_gamma_sweep', action='store_true',
                        help='Skip the gamma sensitivity sweep (faster)')

    args = parser.parse_args()

    print("\n" + "="*66)
    print("  NOP MINI PROJECT - THEME 3")
    print("  Dynamic Soft-Thresholding for Feature Selection")
    print("  in High-Dimensional Regression")
    print("  Jain (Deemed-to-be University) | CSE-AI&ML")
    print("="*66)

    print(f"\nConfiguration:")
    print(f"  Polynomial degree:  {args.poly_degree}")
    print(f"  Base lambda_0:     {args.lambda_0}")
    print(f"  Adaptive gamma:    {args.gamma}")
    print(f"  Max iterations:    {args.max_iter}")
    print(f"  Results directory:  {args.results_dir}")

    # --- Phase 1: Train all models ---
    all_results, data = run_all_experiments(
        poly_degree=args.poly_degree,
        lambda_0=args.lambda_0,
        gamma=args.gamma,
        apg_max_iter=args.max_iter,
        results_dir=args.results_dir,
    )

    # --- Phase 2: Detailed evaluation ---
    eval_report = evaluate_all_models(all_results, data, args.results_dir)

    # --- Phase 3: Gamma sensitivity sweep ---
    gamma_sweep = None
    if not args.skip_gamma_sweep:
        gamma_sweep = run_gamma_sweep(
            poly_degree=args.poly_degree,
            lambda_0=args.lambda_0,
            apg_max_iter=args.max_iter,
            results_dir=args.results_dir,
        )

    # --- Phase 4: Generate plots ---
    plots_dir = os.path.join(args.results_dir, 'plots')
    generate_all_plots(all_results, data, save_dir=plots_dir,
                       gamma_sweep_results=gamma_sweep)

    print("\n" + "="*66)
    print("  PIPELINE COMPLETE!")
    print("="*66)
    print("  Results saved to:")
    print("    - results/comparison_summary.json")
    print("    - results/evaluation_report.json")
    print("    - results/gamma_sweep.json")
    print("    - results/plots/*.png")
    print("")
    print("  To launch the dashboard:")
    print("    python dashboard/app.py")
    print("="*66)


if __name__ == '__main__':
    main()
