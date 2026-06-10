# =============================================================================
# Power Consumption Prediction — FastAPI Web Server
# app.py
# Run: uvicorn app:app --port 8000 --reload
# =============================================================================

import os
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from fastapi.exceptions import RequestValidationError

app = FastAPI(
    title="Power Consumption Prediction API",
    description="Backend API serving performance metrics, feature importance, and machine learning predictions."
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    print("\n[VALIDATION ERROR DETAILS]:")
    print(exc.errors())
    print("Request body received:", exc.body)
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )

@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    # Disable cache to avoid browser caching issues during development
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load model artifacts at startup ──
def load_artifacts():
    required = [
        os.path.join(BASE_DIR, 'models', 'best_model.pkl'),
        os.path.join(BASE_DIR, 'models', 'base_models.pkl'),
        os.path.join(BASE_DIR, 'models', 'scaler.pkl'),
        os.path.join(BASE_DIR, 'models', 'feature_names.pkl'),
        os.path.join(BASE_DIR, 'models', 'metrics.pkl'),
        os.path.join(BASE_DIR, 'models', 'test_results.pkl'),
        os.path.join(BASE_DIR, 'models', 'feature_importance.pkl'),
        os.path.join(BASE_DIR, 'models', 'best_model_name.pkl'),
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        raise RuntimeError(f"Missing model artifacts: {missing}. Run main.py first.")

    artifacts = {
        'best_model':         joblib.load(os.path.join(BASE_DIR, 'models', 'best_model.pkl')),
        'base_models':        joblib.load(os.path.join(BASE_DIR, 'models', 'base_models.pkl')),
        'scaler':             joblib.load(os.path.join(BASE_DIR, 'models', 'scaler.pkl')),
        'feature_names':      joblib.load(os.path.join(BASE_DIR, 'models', 'feature_names.pkl')),
        'metrics':            joblib.load(os.path.join(BASE_DIR, 'models', 'metrics.pkl')),
        'test_results':       joblib.load(os.path.join(BASE_DIR, 'models', 'test_results.pkl')),
        'feature_importance': joblib.load(os.path.join(BASE_DIR, 'models', 'feature_importance.pkl')),
        'best_model_name':    joblib.load(os.path.join(BASE_DIR, 'models', 'best_model_name.pkl')),
    }
    return artifacts

try:
    artifacts = load_artifacts()
    best_model = artifacts['best_model']
    base_models = artifacts['base_models']
    scaler = artifacts['scaler']
    feature_names = artifacts['feature_names']
    metrics = artifacts['metrics']
    test_results = artifacts['test_results']
    feature_importance = artifacts['feature_importance']
    best_model_name = artifacts['best_model_name']
except Exception as e:
    print(f"ERROR: Failed to load model artifacts: {e}")
    # Define placeholder variables so the app can compile if main.py hasn't run yet
    best_model = None
    base_models = {}
    scaler = None
    feature_names = []
    metrics = {}
    test_results = None
    feature_importance = {}
    best_model_name = "None"

# ── Request input validation schema ──
class PredictionInputs(BaseModel):
    temperature: float
    hour: int
    season: int
    voltage: float
    lag_1: float
    lag_24: float
    current_intensity: float

# ── Static File mounts ──
# (Mount plots and static directories if they exist)
if os.path.exists(os.path.join(BASE_DIR, "plots")):
    app.mount("/plots", StaticFiles(directory=os.path.join(BASE_DIR, "plots")), name="plots")

# ── API Endpoints ──

@app.get("/")
def read_root():
    """Serves the main single-page web app."""
    index_path = os.path.join(BASE_DIR, "static", "index.html")
    if not os.path.exists(index_path):
        return {"message": "FastAPI backend is running. Welcome! Place index.html inside 'static' to view UI."}
    return FileResponse(index_path)

@app.get("/api/metrics")
def get_metrics():
    """Returns the model comparison evaluation metrics and best model name."""
    if not metrics:
        raise HTTPException(status_code=500, detail="Metrics not loaded. Run main.py first.")
    return {
        "metrics": metrics,
        "best_model_name": best_model_name
    }

@app.get("/api/feature-importance")
def get_feature_importance():
    """Returns Random Forest feature importance scores."""
    if not isinstance(feature_importance, pd.Series) and not feature_importance:
         raise HTTPException(status_code=500, detail="Feature importance not loaded. Run main.py first.")
    
    fi_list = []
    for rank, (feat, score) in enumerate(feature_importance.items(), 1):
        cum = float(feature_importance.head(rank).sum())
        fi_list.append({
            "rank": rank,
            "feature": feat,
            "importance": float(score),
            "cumulative": cum
        })
    return fi_list

@app.post("/api/predict")
def predict_consumption(inputs: PredictionInputs):
    """Executes prediction scaling, inference, contribution breakdown, and validation comparison."""
    if not best_model or not scaler or not feature_names:
        raise HTTPException(status_code=500, detail="Models/scaler not loaded on server. Run main.py first.")

    # Build input dictionary with plain English labels
    input_dict = {
        'Temperature (°C)':                 inputs.temperature,
        'Hour of Day':                      inputs.hour,
        'Season':                           inputs.season,
        'Voltage (V)':                      inputs.voltage,
        'Consumption 1 Hour Ago (kW)':      inputs.lag_1,
        'Consumption 24 Hours Ago (kW)':    inputs.lag_24,
        'Current Intensity (A)':            inputs.current_intensity,
    }

    # Format input array ordered by feature_names
    try:
        input_row = np.array([input_dict[feat] for feat in feature_names]).reshape(1, -1)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing expected model feature: {e}")

    # Scale the inputs
    input_scaled = scaler.transform(input_row)

    # Base model predictions
    base_preds = {}
    for name, model in base_models.items():
        base_preds[name] = float(model.predict(input_scaled)[0])

    # Best model prediction
    prediction = float(best_model.predict(input_scaled)[0])

    # Extract weights & intercept for Selected Model (Linear Regression)
    lr_model = base_models.get('Linear Regression')
    if lr_model is None:
        raise HTTPException(status_code=500, detail="Linear Regression base model not found.")
    
    intercept = float(lr_model.intercept_)
    coefs = lr_model.coef_

    # Calculate step-by-step contributions
    calc_steps = []
    sum_contributions = 0.0
    for idx, feat in enumerate(feature_names):
        raw_val = input_dict[feat]
        mean_val = scaler.mean_[idx]
        std_val = scaler.scale_[idx]
        scaled_val = (raw_val - mean_val) / std_val
        coef_val = coefs[idx]
        contrib = scaled_val * coef_val
        sum_contributions += contrib

        calc_steps.append({
            "feature": feat,
            "raw_value": float(raw_val),
            "mean": float(mean_val),
            "std": float(std_val),
            "scaled": float(scaled_val),
            "weight": float(coef_val),
            "contribution": float(contrib)
        })

    # Build LaTeX formula equation
    coef_terms = []
    for idx, feat in enumerate(feature_names):
        coef_val = coefs[idx]
        sign = "+" if coef_val >= 0 else "-"
        # Shorten feature name label for equation
        short_feat = (feat
                      .replace(" (V)", "")
                      .replace(" (A)", "")
                      .replace(" (Kitchen Wh)", "")
                      .replace(" (Laundry Wh)", "")
                      .replace(" (°C)", "")
                      .replace(" (km/h)", "")
                      .replace(" (mm)", "")
                      .replace("Consumption 1 Hour Ago (kW)", "Lag1")
                      .replace("Consumption 2 Hours Ago (kW)", "Lag2")
                      .replace("Consumption 3 Hours Ago (kW)", "Lag3")
                      .replace("Consumption 24 Hours Ago (kW)", "Lag24")
                     )
        coef_terms.append(f"{sign} {abs(coef_val):.4f} X_{{\\text{{{short_feat}}}}}")
    
    equation_latex = f"\\text{{Predicted kW}} = {intercept:.4f} " + " ".join(coef_terms)

    # Draw 5 comparison validation samples from test set
    validation_samples = []
    if test_results is not None:
        np.random.seed(42) # Consistent display rows
        sample_indices = np.random.choice(len(test_results), 5, replace=False)
        sample_df = test_results.iloc[sample_indices]
        for _, row in sample_df.iterrows():
            dt_str = pd.to_datetime(row['Date']).strftime('%b %d, %Y, %H:%M')
            validation_samples.append({
                "date": dt_str,
                "actual": float(row['Electricity Consumption (kW)']),
                "predicted": float(row['Linear Regression']),
                "error": float(abs(row['Electricity Consumption (kW)'] - row['Linear Regression']))
            })

    return JSONResponse(content={
        "prediction": prediction,
        "base_predictions": base_preds,
        "intercept": intercept,
        "sum_contributions": float(sum_contributions),
        "sum_pred": float(intercept + sum_contributions),
        "formula_latex": equation_latex,
        "calculation_steps": calc_steps,
        "validation_samples": validation_samples
    })

# Mount static folder for CSS, JS, images
if os.path.exists(os.path.join(BASE_DIR, "static")):
    app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
