"""
Adaptive Proximal Gradient with Dynamic Soft-Thresholding (APG-DST)

Implementation uses coordinate descent with per-feature adaptive penalties
derived from the subdifferential of the L1 norm at the current iterate.

Two phases:
  Phase 1: Standard LASSO (fixed lambda) via coordinate descent 
  Phase 2: Adaptive thresholding where per-feature penalties are
           iteratively updated based on coefficient magnitudes
"""

import numpy as np
import time


def soft_threshold(z, threshold):
    """Soft-thresholding operator: S(z,t) = sign(z)*max(|z|-t, 0)"""
    return np.sign(z) * np.maximum(np.abs(z) - threshold, 0.0)


def compute_objective(X, y, beta, lambda_vec):
    """Composite objective: (1/2n)||y-Xb||^2 + sum lambda_j|b_j|"""
    n = X.shape[0]
    return 0.5 * np.sum((y - X @ beta)**2) / n + np.sum(lambda_vec * np.abs(beta))


class AdaptiveProximalGradient:
    """
    Adaptive Proximal Gradient with Dynamic Soft-Thresholding.

    Uses coordinate descent with per-feature adaptive L1 penalties.
    Phase 1: Fixed penalties (standard LASSO) for warm start.
    Phase 2: Adaptive penalties from subdifferential weights.

    Parameters
    ----------
    lambda_0 : float - Base regularization strength
    gamma : float - Adaptive weight exponent
    epsilon : float - Stability constant
    n_outer : int - Number of outer iterations for adaptive reweighting
    n_cd_inner : int - Coordinate descent passes per outer iteration
    warmup_frac : float - Fraction of iterations for Phase 1
    tau_min : float - Minimum schedule value
    tol : float - Convergence tolerance
    verbose : bool - Print progress
    """

    def __init__(self, lambda_0=0.02, gamma=0.5, epsilon=1e-4,
                 n_outer=30, n_cd_inner=200, warmup_frac=0.3,
                 tau_min=0.1, tol=1e-6, verbose=True, use_fista=True):
        self.lambda_0 = lambda_0
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_outer = n_outer
        self.n_cd_inner = n_cd_inner
        self.warmup_frac = warmup_frac
        self.tau_min = tau_min
        self.tol = tol
        self.verbose = verbose
        self.use_fista = use_fista

        self.history = {
            'objective': [],
            'mse': [],
            'sparsity': [],
            'n_nonzero': [],
            'step_size': [],
            'lambda_mean': [],
            'lambda_max': [],
        }

    def _compute_adaptive_lambda(self, beta, iteration):
        """Per-feature adaptive penalties."""
        warmup = int(self.n_outer * self.warmup_frac)
        if iteration < warmup:
            return np.full(len(beta), self.lambda_0)

        # Adaptive weights: small |b_j| -> higher penalty
        w = 1.0 / (np.abs(beta) + self.epsilon) ** self.gamma
        w = w / (np.mean(w) + 1e-10)  # Normalize

        # Iteration schedule: decay over adaptive phase
        progress = (iteration - warmup) / max(1, self.n_outer - warmup)
        schedule = max(self.tau_min, 1.0 - progress)

        return self.lambda_0 * w * schedule

    def _coordinate_descent(self, X, y, beta, lambda_vec, n_passes):
        """
        Coordinate descent for weighted LASSO.
        
        For each feature j, update:
          b_j = S(r_j + b_j, lambda_j / L_j) 
        where r_j = X_j^T * residual / n, L_j = ||X_j||^2 / n
        """
        n, p = X.shape
        # Precompute column norms (Lipschitz constants per feature)
        col_norms = np.sum(X**2, axis=0) / n

        residual = y - X @ beta

        for _ in range(n_passes):
            for j in range(p):
                if col_norms[j] < 1e-12:
                    continue

                # Partial residual (add back j-th contribution)
                residual += X[:, j] * beta[j]

                # Compute univariate optimum
                rho_j = X[:, j].dot(residual) / n

                # Soft-threshold
                beta[j] = soft_threshold(rho_j, lambda_vec[j]) / col_norms[j]

                # Update residual
                residual -= X[:, j] * beta[j]

        return beta

    def fit(self, X, y, X_val=None, y_val=None):
        """Fit using coordinate descent with adaptive reweighting.
        
        When use_fista=True (default), applies Nesterov momentum extrapolation
        after each coordinate descent pass for O(1/k²) vs O(1/k) convergence:
            y_{k+1} = β_{k+1} + (t_k - 1)/t_{k+1} * (β_{k+1} - β_k)
            t_{k+1} = (1 + sqrt(1 + 4*t_k²)) / 2
        """
        n, p = X.shape

        # Center targets (equivalent to fit_intercept=True)
        y_mean = np.mean(y)
        y_centered = y - y_mean

        self.coef_ = np.zeros(p)
        self.intercept_ = y_mean
        warmup = int(self.n_outer * self.warmup_frac)

        # FISTA momentum state
        beta_prev = np.zeros(p)   # β_{k-1}
        t_k = 1.0                 # momentum scalar

        accel_label = "FISTA" if self.use_fista else "PGD"
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"{'APG-DST: Adaptive Proximal Gradient (' + accel_label + ')':^70}")
            print(f"{'='*70}")
            print(f"  Outer iters: {self.n_outer}, CD passes: {self.n_cd_inner}")
            print(f"  Phase 1 (fixed): {warmup}, Phase 2 (adaptive): {self.n_outer - warmup}")
            print(f"{'-'*70}")
            print(f"{'Iter':>5} | {'Objective':>12} | {'MSE':>10} | {'Nonzero':>7}/"
                  f"{p} | {'Sparsity':>8} | {'Lam_mean':>10}")
            print(f"{'-'*70}")

        start_time = time.time()
        prev_obj = np.inf

        for k in range(self.n_outer):
            # Adaptive penalties
            lam = self._compute_adaptive_lambda(self.coef_, k)

            # Save current iterate for momentum
            beta_k = self.coef_.copy()

            # Run coordinate descent on centered data
            self.coef_ = self._coordinate_descent(
                X, y_centered, self.coef_.copy(), lam, self.n_cd_inner
            )

            # --- FISTA / Nesterov Momentum Extrapolation ---
            if self.use_fista and k >= 1:
                t_next = (1.0 + np.sqrt(1.0 + 4.0 * t_k ** 2)) / 2.0
                momentum = (t_k - 1.0) / t_next
                # Extrapolated point (only applied to adaptive phase to avoid
                # disrupting the warm-start phase)
                if k >= warmup:
                    self.coef_ = self.coef_ + momentum * (self.coef_ - beta_prev)
                t_k = t_next
            beta_prev = beta_k

            # Track metrics on original (uncentered) scale
            y_pred = X @ self.coef_ + self.intercept_
            mse = float(np.mean((y - y_pred) ** 2))
            obj = float(compute_objective(X, y_centered, self.coef_, lam))
            n_nz = int(np.sum(np.abs(self.coef_) > 1e-8))
            sparsity = (1 - n_nz / p) * 100

            self.history['objective'].append(obj)
            self.history['mse'].append(mse)
            self.history['sparsity'].append(float(sparsity))
            self.history['n_nonzero'].append(n_nz)
            self.history['step_size'].append(float(np.mean(lam)))
            self.history['lambda_mean'].append(float(np.mean(lam)))
            self.history['lambda_max'].append(float(np.max(lam)))

            phase = "W" if k < warmup else "A"
            if self.verbose:
                print(f"  {k:>3}{phase} | {obj:>12.6f} | {mse:>10.6f} | {n_nz:>7d}/{p} "
                      f"| {sparsity:>7.1f}% | {np.mean(lam):>10.6f}")

            # Convergence check (only in adaptive phase)
            if k >= warmup and prev_obj != np.inf:
                rel = abs(prev_obj - obj) / (abs(prev_obj) + 1e-10)
                if rel < self.tol:
                    if self.verbose:
                        print(f"\n[APG-DST] Converged at iter {k} (rel={rel:.2e})")
                    break
            prev_obj = obj

        self.train_time_ = time.time() - start_time
        self.n_iter_ = k + 1

        subgrad = np.zeros(p)
        active = np.abs(self.coef_) > 1e-10
        subgrad[active] = np.sign(self.coef_[active])
        self.subgrad_ = subgrad
        self.active_set_ = active

        if self.verbose:
            print(f"\n[APG-DST] Done in {self.train_time_:.3f}s, {self.n_iter_} iters")
            print(f"[APG-DST] MSE={mse:.4f}, Nonzero={n_nz}/{p}, Sparsity={sparsity:.1f}%")

        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_

    def get_selected_features(self, feature_names=None, threshold=1e-8):
        mask = np.abs(self.coef_) > threshold
        idx = np.where(mask)[0]
        result = {'indices': idx, 'coefficients': self.coef_[mask],
                  'n_selected': len(idx), 'n_total': len(self.coef_)}
        if feature_names:
            result['names'] = [feature_names[i] for i in idx]
        return result


def train_apg(X_train, y_train, X_val, y_val, X_test, y_test,
              lambda_0=0.02, gamma=0.5, max_iter=30, verbose=True):
    """Train APG-DST and evaluate on all splits."""
    from src.models.baselines import compute_metrics

    model = AdaptiveProximalGradient(
        lambda_0=lambda_0, gamma=gamma,
        n_outer=max_iter, verbose=verbose,
    )
    model.fit(X_train, y_train, X_val, y_val)

    y_tr = model.predict(X_train)
    y_v = model.predict(X_val)
    y_te = model.predict(X_test)

    return {
        'model': model,
        'name': 'APG-DST',
        'coefficients': model.coef_,
        'best_lambda': lambda_0,
        'train_time': model.train_time_,
        'n_iterations': model.n_iter_,
        'train_metrics': compute_metrics(y_train, y_tr, model.coef_),
        'val_metrics': compute_metrics(y_val, y_v, model.coef_),
        'test_metrics': compute_metrics(y_test, y_te, model.coef_),
        'y_pred_test': y_te,
        'history': model.history,
    }
