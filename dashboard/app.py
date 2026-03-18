"""
Flask Web Dashboard for visualizing experiment results.

Displays training results, convergence plots, feature selection analysis,
model comparison, and real-time price prediction in an interactive web interface.

Usage:
    python dashboard/app.py
    Then open http://localhost:8000 in your browser.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, send_from_directory, request

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.datasets import fetch_california_housing

app = Flask(__name__)

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')

# --- Feature names for the California Housing dataset ---
FEATURE_NAMES = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                 'Population', 'AveOccup', 'Latitude', 'Longitude']

FEATURE_INFO = {
    'MedInc':     {'label': 'Median Income',               'placeholder': '3.5',    'unit': '(tens of thousands $)', 'min': 0.5, 'max': 15.0, 'step': 0.1},
    'HouseAge':   {'label': 'House Age',                   'placeholder': '28',     'unit': '(years)',               'min': 1,   'max': 52,   'step': 1},
    'AveRooms':   {'label': 'Average Rooms',               'placeholder': '5.4',    'unit': '(per household)',       'min': 0.5, 'max': 50.0, 'step': 0.1},
    'AveBedrms':  {'label': 'Average Bedrooms',            'placeholder': '1.1',    'unit': '(per household)',       'min': 0.3, 'max': 30.0, 'step': 0.1},
    'Population': {'label': 'Population',                  'placeholder': '1200',   'unit': '(block group)',         'min': 3,   'max': 40000,'step': 1},
    'AveOccup':   {'label': 'Average Occupancy',           'placeholder': '3.0',    'unit': '(per household)',       'min': 0.5, 'max': 1200, 'step': 0.1},
    'Latitude':   {'label': 'Latitude',                    'placeholder': '34.05',  'unit': '(degrees)',             'min': 32.5,'max': 42.0, 'step': 0.01},
    'Longitude':  {'label': 'Longitude',                   'placeholder': '-118.25','unit': '(degrees)',             'min': -124.5,'max': -114.0,'step': 0.01},
}

# --- Global objects for prediction pipeline (loaded once at startup) ---
_poly = None
_scaler = None
_models = {}  # {'Ridge': (coef, intercept), 'LASSO': (coef, intercept), 'APG-DST': (coef, intercept)}


def _init_prediction_pipeline():
    """Load dataset to fit scaler/poly, and train fast models on startup."""
    global _poly, _scaler, _models

    print("[Predict] Initializing prediction pipeline...")

    # Load the California Housing data
    housing = fetch_california_housing()
    X_raw = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = housing.target.copy()

    # Polynomial expansion (same as dataset.py)
    _poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    X_poly = _poly.fit_transform(X_raw)

    # Fit the scaler on the full dataset (close-enough approximation)
    _scaler = StandardScaler()
    _scaler.fit(X_poly)

    # Try to load saved coefficients
    coef_files = {
        'Ridge':   os.path.join(RESULTS_DIR, 'ridge_coefficients.npy'),
        'LASSO':   os.path.join(RESULTS_DIR, 'lasso_coefficients.npy'),
        'APG-DST': os.path.join(RESULTS_DIR, 'apg_dst_coefficients.npy'),
    }

    X_scaled = _scaler.transform(X_poly)
    y_mean = float(np.mean(y))

    for name, path in coef_files.items():
        if os.path.exists(path):
            coef = np.load(path)
            # Compute the intercept: y_mean - X_mean_scaled @ coef
            # Since scaler centers to 0, X_mean_scaled ≈ 0 → intercept ≈ y_mean
            # But to be more precise, use the actual scaler mean
            X_mean_scaled = _scaler.transform(_poly.transform(X_raw.mean().values.reshape(1, -1))).flatten()
            intercept = y_mean - X_mean_scaled @ coef
            _models[name] = (coef, intercept)
            print(f"[Predict] Loaded {name} coefficients ({len(coef)} features)")
        else:
            print(f"[Predict] Warning: {path} not found, {name} predictions unavailable")

    print(f"[Predict] Pipeline ready — {len(_models)} model(s) loaded")


def predict_price(features_dict):


    if _poly is None or _scaler is None:
        return {}

    # Build a single-row DataFrame
    row = pd.DataFrame([features_dict], columns=FEATURE_NAMES)

    # Polynomial expansion
    X_poly = _poly.transform(row)

    # Scale
    X_scaled = _scaler.transform(X_poly)

    predictions = {}
    for name, (coef, intercept) in _models.items():
        pred = float(X_scaled @ coef + intercept)
        predictions[name] = round(pred, 4)

    return predictions


def load_json(filename):
    """Load a JSON file from the results directory."""
    filepath = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


@app.route('/')
def home():
    """Home page with project overview and summary metrics."""
    summary = load_json('comparison_summary.json')
    eval_report = load_json('evaluation_report.json')
    return render_template('home.html', summary=summary, eval_report=eval_report)


@app.route('/convergence')
def convergence():
    """APG-DST convergence analysis page."""
    history = load_json('apg_history.json')
    return render_template('convergence.html', history=history)


@app.route('/features')
def features():
    """Feature selection analysis page."""
    eval_report = load_json('evaluation_report.json')
    return render_template('features.html', eval_report=eval_report)


@app.route('/comparison')
def comparison():
    """Model comparison page."""
    summary = load_json('comparison_summary.json')
    eval_report = load_json('evaluation_report.json')
    return render_template('comparison.html', summary=summary, eval_report=eval_report)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Price prediction page — user inputs features and gets model predictions."""
    predictions = None
    input_values = {}
    error = None

    if request.method == 'POST':
        try:
            input_values = {}
            for feat in FEATURE_NAMES:
                val = request.form.get(feat, '').strip()
                if not val:
                    raise ValueError(f"Please fill in the {FEATURE_INFO[feat]['label']} field.")
                input_values[feat] = float(val)

            predictions = predict_price(input_values)

            if not predictions:
                error = "No trained models found. Run 'python main.py' first to generate model coefficients."
        except ValueError as e:
            error = str(e)

    return render_template(
        'predict.html',
        feature_names=FEATURE_NAMES,
        feature_info=FEATURE_INFO,
        predictions=predictions,
        input_values=input_values,
        error=error,
    )


@app.route('/plots/<filename>')
def serve_plot(filename):
    """Serve plot images."""
    return send_from_directory(PLOTS_DIR, filename)


if __name__ == '__main__':
    _init_prediction_pipeline()
    print("\n  Dashboard running at http://localhost:8000")
    print("  Press Ctrl+C to stop.\n")
    app.run(debug=True, port=8000, use_reloader=False)
