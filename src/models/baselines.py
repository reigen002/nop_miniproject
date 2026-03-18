"""
Baseline regression models: Ridge and standard LASSO.

These serve as comparison baselines for the custom Adaptive Proximal
Gradient optimizer with dynamic soft-thresholding.
"""

import numpy as np
import time
from sklearn.linear_model import Ridge, Lasso, LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def compute_metrics(y_true, y_pred, coefficients=None):
    """Compute standard regression metrics and sparsity statistics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    result = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
    }

    if coefficients is not None:
        n_features = len(coefficients)
        n_zero = np.sum(np.abs(coefficients) < 1e-8)
        n_nonzero = n_features - n_zero
        sparsity = n_zero / n_features * 100

        result.update({
            'n_features': n_features,
            'n_nonzero': n_nonzero,
            'n_zero': n_zero,
            'sparsity_pct': sparsity,
        })

    return result


def train_ridge(X_train, y_train, X_val, y_val, X_test, y_test, alphas=None):
    """
    Train Ridge regression with cross-validated alpha selection.

    Returns
    -------
    dict with model, coefficients, metrics on all splits, and timing info.
    """
    if alphas is None:
        alphas = np.logspace(-3, 3, 50)

    print("[Ridge] Training with cross-validated alpha selection...")
    start_time = time.time()

    model = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error')
    model.fit(X_train, y_train)

    train_time = time.time() - start_time

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    coef = model.coef_

    results = {
        'model': model,
        'name': 'Ridge',
        'coefficients': coef,
        'best_alpha': model.alpha_,
        'train_time': train_time,
        'train_metrics': compute_metrics(y_train, y_pred_train, coef),
        'val_metrics': compute_metrics(y_val, y_pred_val, coef),
        'test_metrics': compute_metrics(y_test, y_pred_test, coef),
        'y_pred_test': y_pred_test,
    }

    print(f"[Ridge] Best alpha: {model.alpha_:.6f}")
    print(f"[Ridge] Train MSE: {results['train_metrics']['mse']:.4f}, "
          f"Val MSE: {results['val_metrics']['mse']:.4f}, "
          f"Test MSE: {results['test_metrics']['mse']:.4f}")
    print(f"[Ridge] Sparsity: {results['test_metrics']['sparsity_pct']:.1f}% zero coefficients")
    print(f"[Ridge] Training time: {train_time:.3f}s")

    return results


def train_lasso(X_train, y_train, X_val, y_val, X_test, y_test, alphas=None, max_iter=10000):
    """
    Train standard LASSO regression with cross-validated alpha selection.

    Returns
    -------
    dict with model, coefficients, metrics, convergence history, and timing.
    """
    if alphas is None:
        alphas = np.logspace(-4, 1, 50)

    print("[LASSO] Training with cross-validated alpha selection...")
    start_time = time.time()

    model = LassoCV(alphas=alphas, cv=5, max_iter=max_iter, random_state=42)
    model.fit(X_train, y_train)

    train_time = time.time() - start_time

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    coef = model.coef_

    results = {
        'model': model,
        'name': 'LASSO',
        'coefficients': coef,
        'best_alpha': model.alpha_,
        'train_time': train_time,
        'train_metrics': compute_metrics(y_train, y_pred_train, coef),
        'val_metrics': compute_metrics(y_val, y_pred_val, coef),
        'test_metrics': compute_metrics(y_test, y_pred_test, coef),
        'y_pred_test': y_pred_test,
    }

    print(f"[LASSO] Best alpha: {model.alpha_:.6f}")
    print(f"[LASSO] Train MSE: {results['train_metrics']['mse']:.4f}, "
          f"Val MSE: {results['val_metrics']['mse']:.4f}, "
          f"Test MSE: {results['test_metrics']['mse']:.4f}")
    print(f"[LASSO] Sparsity: {results['test_metrics']['sparsity_pct']:.1f}% zero coefficients")
    print(f"[LASSO] Training time: {train_time:.3f}s")

    return results


def train_elastic_net(X_train, y_train, X_val, y_val, X_test, y_test,
                      alphas=None, l1_ratios=None, max_iter=10000):
    """
    Train Elastic Net regression with cross-validated alpha and l1_ratio selection.
    Elastic Net combines L1 (LASSO) and L2 (Ridge) penalties:
        min (1/2n)||y-Xβ||² + α*l1_ratio*||β||₁ + α*(1-l1_ratio)/2*||β||₂²

    Returns
    -------
    dict with model, coefficients, metrics on all splits, and timing info.
    """
    if alphas is None:
        alphas = np.logspace(-4, 1, 30)
    if l1_ratios is None:
        l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]

    print("[ElasticNet] Training with cross-validated alpha/l1_ratio selection...")
    start_time = time.time()

    model = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios, cv=5,
                         max_iter=max_iter, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    train_time = time.time() - start_time

    y_pred_train = model.predict(X_train)
    y_pred_val   = model.predict(X_val)
    y_pred_test  = model.predict(X_test)

    coef = model.coef_

    results = {
        'model':         model,
        'name':          'ElasticNet',
        'coefficients':  coef,
        'best_alpha':    model.alpha_,
        'best_l1_ratio': model.l1_ratio_,
        'train_time':    train_time,
        'train_metrics': compute_metrics(y_train, y_pred_train, coef),
        'val_metrics':   compute_metrics(y_val,   y_pred_val,   coef),
        'test_metrics':  compute_metrics(y_test,  y_pred_test,  coef),
        'y_pred_test':   y_pred_test,
    }

    print(f"[ElasticNet] Best alpha: {model.alpha_:.6f}, l1_ratio: {model.l1_ratio_:.2f}")
    print(f"[ElasticNet] Train MSE: {results['train_metrics']['mse']:.4f}, "
          f"Val MSE: {results['val_metrics']['mse']:.4f}, "
          f"Test MSE: {results['test_metrics']['mse']:.4f}")
    print(f"[ElasticNet] Sparsity: {results['test_metrics']['sparsity_pct']:.1f}% zero coefficients")
    print(f"[ElasticNet] Training time: {train_time:.3f}s")

    return results
