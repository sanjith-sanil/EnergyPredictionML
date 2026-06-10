# ⚡ Power Consumption Prediction using Machine Learning and Smart Meter Data

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-orange?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-red?logo=streamlit&logoColor=white)

---

## Overview

This project evaluates three machine learning models (Linear Regression, Random Forest, and Gradient Boosting)
to predict daily household electricity consumption using smart meter readings and weather data, selecting the
best individual model (Linear Regression) for production. The selected model achieves robust predictions (R² = 0.9992)
that account for seasonal patterns, weather effects, and past consumption behavior.

---

## Objectives

1. Develop a hybrid machine learning model for accurate prediction of home energy consumption using smart meter data.
2. Analyze the impact of factors such as time, weather, and user behavior on energy usage patterns.
3. Improve prediction accuracy using appropriate feature selection and model optimization techniques.

---

## Dataset Description

| Dataset | Source | Size | Date Range | Key Columns |
|---------|--------|------|------------|-------------|
| **household_power_consumption.csv** | UCI ML Repository | ~2 million rows (minute-level) | Dec 2006 – Dec 2008 | `Global_active_power`, `Global_reactive_power`, `Voltage`, `Global_intensity`, `Sub_metering_1/2/3` |
| **sceaux_weather_data.csv** | Meteostat | ~1,400 rows (daily) | Dec 2006 – Nov 2010 | `tavg`, `tmin`, `tmax`, `prcp`, `wspd`, `wdir`, `pres` |

After merging on date and feature engineering, the usable dataset is approximately **700 daily rows** (Dec 2006 – Dec 2008 overlap).

---

## Project Structure

```
project/
├── main.py               # Full ML pipeline — 8 sections (load, clean, engineer, select, tune, train, evaluate, save)
├── app.py                # Streamlit front end — 4 pages (Overview, EDA, Model Performance, Predict)
├── README.md             # Project documentation (this file)
├── requirements.txt      # All Python dependencies pinned to specific versions
├── plots/                # Auto-created by main.py — 7 PNG plots
│   ├── actual_vs_predicted.png
│   ├── residuals.png
│   ├── feature_importance.png
│   ├── eda_time_series.png
│   ├── eda_distribution.png
│   ├── correlation_heatmap.png
│   └── eda_seasonal.png
└── models/               # Auto-created by main.py — 8 PKL artifacts
    ├── best_model.pkl
    ├── base_models.pkl
    ├── scaler.pkl
    ├── feature_names.pkl
    ├── metrics.pkl
    ├── test_results.pkl
    ├── feature_importance.pkl
    └── best_model_name.pkl
```

---

## Model Selection Architecture

The project trains three diverse machine learning models and programmatically selects the best-performing individual model for final production use:

### Evaluated Models
Three diverse regressors are trained on the training set and optimized using time-series cross-validation:
- **Linear Regression** — fast, interpretable model representing linear relationships (selected for production).
- **Random Forest** — captures nonlinear patterns and feature interactions.
- **Gradient Boosting** — sequential decision tree ensemble that optimizes prediction residuals.

### Why Linear Regression was Selected
Because the dataset includes electrical grid variables like `Current Intensity (A)`, the active power has an almost perfectly linear physical relationship ($P \approx V \times I$). Linear Regression fits this physical law directly, achieving an R² of **0.9992** on the test set, outperforming tree-based approximations and stacking ensembling methods which introduce step-function noise.

### Selection Flow (ASCII)

```
                ┌─────────────────────────────────────────────────┐
                │             INPUT FEATURES                      │
                │         (Plain English Labels)                  │
                └──────────────┬──────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
   ┌──────▼──────┐    ┌────────▼───────┐    ┌──────▼──────────┐
   │   Linear    │    │    Random      │    │    Gradient     │
   │ Regression  │    │    Forest      │    │    Boosting     │
   └──────┬──────┘    └────────────────┘    └─────────────────┘
          │ (Selected Best Model: R² = 0.9992)
          │
          └────────────────────┬────────────────────┘
                               │
                   ┌───────────▼───────────┐
                   │  Predicted Electricity │
                   │   Consumption (kW)    │
                   └───────────────────────┘
```

---

## Feature Engineering

All features used in model training use **plain English labels** (Layer 2 naming convention):

| Plain English Label | Original Column | Description |
|---------------------|-----------------|-------------|
| **Yesterday's Consumption (kW)** | `lag_1` | Primary autocorrelation driver — yesterday's usage strongly predicts today's |
| **2 Days Ago Consumption (kW)** | `lag_2` | Short-term usage pattern reinforcement |
| **3 Days Ago Consumption (kW)** | `lag_3` | Captures 3-day usage trends |
| **Last Week's Consumption (kW)** | `lag_7` | Weekly behavioral cycles (e.g., weekday vs. weekend routines) |
| **7-Day Average Consumption (kW)** | `rolling_mean_7` | Smoothed short-term trend — reduces noise |
| **30-Day Average Consumption (kW)** | `rolling_mean_30` | Long-term seasonal baseline |
| **Average Temperature (°C)** | `tavg` | Core weather feature; heating/cooling demand driver |
| **Minimum Temperature (°C)** | `tmin` | Cold extremes trigger heating spikes |
| **Maximum Temperature (°C)** | `tmax` | Hot extremes drive cooling demand |
| **Rainfall (mm)** | `prcp` | Indoor activity proxy on rainy days |
| **Wind Speed (km/h)** | `wspd` | Affects heating losses and wind-chill perception |
| **Wind Direction (°)** | `wdir` | Secondary weather signal |
| **Atmospheric Pressure (hPa)** | `pres` | Weather system indicator |
| **Voltage (V)** | `Voltage` | Electrical grid characteristic |
| **Current Intensity (A)** | `Global_intensity` | Direct electrical load measure |
| **Day of Month** | `day` | Day of the month (1-31) representing monthly behavior cycle |
| **Day of Week** | `day_of_week` | Weekday vs. weekend behavioral patterns |
| **Is Weekend** | `is_weekend` | Binary flag for weekend consumption patterns |
| **Month** | `month` | Captures within-year seasonality |
| **Season** | `season` | 0=Winter, 1=Spring, 2=Summer, 3=Autumn |
| **Quarter** | `quarter` | Coarse seasonal grouping |

> **Note:** `Sub_metering_1`, `Sub_metering_2`, `Sub_metering_3` are excluded (data leakage — they are components of the target). `snow`, `wpgt`, `tsun` are dropped (near-empty).

---

## Installation

```bash
git clone <repo>
cd project
pip install -r requirements.txt
```

---

## How to Run

**Step 1 — Train models and generate artifacts:**
```bash
python main.py
```

This will:
- Load and clean both datasets
- Engineer all features
- Run hyperparameter tuning (RandomizedSearchCV, n_iter=50, ~5–15 min)
- Train base models, evaluate them, and select the best one
- Save model artifacts to `models/` and plots to `plots/`

**Step 2 — Launch the Streamlit front end:**
```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## Results

| Model | MAE | RMSE | R² | MAPE (%) |
|-------|-----|------|-----|----------|
| **Linear Regression (Best Model)** | **0.0117** | **0.0142** | **0.9992** | **2.24%** |
| Random Forest | 0.0713 | 0.1221 | 0.9398 | 24.80% |
| Gradient Boosting | 0.0197 | 0.0260 | 0.9973 | 5.29% |

**Production Model: Linear Regression achieved an R² of 0.9992.**

---

## Screenshots

> Add screenshots of the Streamlit app here after running:
>
> - `plots/actual_vs_predicted.png` — test set predictions overlay
> - `plots/feature_importance.png` — ranked feature contributions
> - App Page 1 — Overview (R² callout, model selection architecture diagram)
> - App Page 2 — Exploratory Data Analysis (Time Series, Distribution with KDE, Heatmap, Seasonal Trend)
> - App Page 4 — Predict Consumption (Input form, dynamic formula walkthrough, calculation breakdown table, test comparison validation)

---

## Technologies Used

| Library | Version | Purpose |
|---------|---------|---------|
| `pandas` | 2.2.2 | Data loading, cleaning, resampling, merging |
| `numpy` | 1.26.4 | Numerical operations, array manipulation |
| `scikit-learn` | 1.4.2 | ML models, feature selection, cross-validation, metrics |
| `matplotlib` | 3.8.4 | Plot generation (saved as PNG) |
| `seaborn` | 0.13.2 | Correlation heatmap visualization |
| `streamlit` | 1.35.0 | Interactive web front end |
| `joblib` | 1.4.2 | Model serialization / artifact persistence |

---

## License

MIT License — free to use and modify with attribution.
