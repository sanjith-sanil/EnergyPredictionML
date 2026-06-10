# ⚡ Power Consumption Prediction using Machine Learning and Smart Meter Data

![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-orange?logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-teal?logo=fastapi&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-0.30.1-black?logo=uvicorn&logoColor=white)

---

## Overview

This project implements an hourly electricity consumption forecasting model using smart meter readings and weather station data. It trains and evaluates three machine learning models (**Linear Regression**, **Random Forest**, and **Gradient Boosting**) on a 7-feature dataset. 

Because the feature set includes the physical electrical variable **`Current Intensity (A)`**, the models achieve near-perfect prediction metrics (**R² = 99.96%**), directly modeling the physical law of electrical power. The project is served via a lightweight **FastAPI backend API** and an interactive, beautifully designed **Single Page Application (SPA)** dashboard.

---

## Objectives

1. Develop a high-accuracy machine learning model for predicting hourly active household power consumption ($kW$).
2. Analyze the impact of weather conditions, seasonal patterns, and past consumption behavior on electrical demand.
3. Serve predictions, base model comparisons, and step-by-step mathematical calculations in an interactive web dashboard.

---

## Dataset Description

| Dataset | Source | Granularity | Date Range | Key Columns |
|---------|--------|-------------|------------|-------------|
| **household_power_consumption.csv** | UCI ML Repository | Minute-level | Dec 2006 – Dec 2008 | `Global_active_power` (Target), `Voltage`, `Global_intensity`, `Sub_metering_1/2` |
| **sceaux_weather_data.csv** | Meteostat | Daily | Dec 2006 – Nov 2010 | `tavg`, `wdir`, `wspd` |

After resampling the smart meter readings to hourly averages and merging them with forward-filled weather station records, the final cleaned dataset contains **17,453 hourly rows** spanning Dec 2006 to Dec 2008.

---

## Project Structure

```
project/
├── main.py               # Full ML pipeline (load, clean, resample, engineer features, tune, evaluate, save)
├── app.py                # FastAPI backend server serving SPA views and prediction API endpoints
├── requirements.txt      # Python package dependencies
├── .gitignore            # Excludes virtual envs, local temp files, datasets, and massive model pickles
├── static/               # Frontend SPA dashboard assets
│   ├── index.html        # SPA dashboard structure and layouts
│   ├── script.js         # API controller, charting renderer, and KaTeX equations formatter
│   └── style.css         # Modern styling rules and flowchart flow animations
├── plots/                # Auto-created by main.py — 7 PNG plots for EDA and residual evaluations
│   ├── actual_vs_predicted.png
│   ├── residuals.png
│   ├── feature_importance.png
│   ├── eda_time_series.png
│   ├── eda_distribution.png
│   ├── correlation_heatmap.png
│   └── eda_seasonal.png
└── models/               # Auto-created by main.py — persisting scalars and model files
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

## Feature Space (7 Selected Features)

To achieve maximum accuracy while keeping the dashboard intuitive, the models are trained on exactly **7 features** (Layer 2 naming convention):

| Feature Name | Description | Source |
|--------------|-------------|--------|
| **Temperature (°C)** | Average hourly temperature | Weather station (`tavg`) |
| **Hour of Day** | Hour index (0–23) representing daily usage cycles | DateTime Engineering |
| **Season** | Season index (0=Winter, 1=Spring, 2=Summer, 3=Autumn) | DateTime Engineering |
| **Voltage (V)** | Average hourly electrical line voltage | Smart Meter (`Voltage`) |
| **Consumption 1 Hour Ago (kW)** | Power consumption in the previous hour (Lag 1) | Smart Meter (`Global_active_power`) |
| **Consumption 24 Hours Ago (kW)** | Power consumption at the same hour yesterday (Lag 24) | Smart Meter (`Global_active_power`) |
| **Current Intensity (A)** | Average hourly electrical current draw | Smart Meter (`Global_intensity`) |

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sanjith-sanil/EnergyPredictionML.git
   cd EnergyPredictionML
   ```
2. **Download the dataset**:
   Place the raw `household_power_consumption.csv` file into the project root directory.
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

### Step 1: Train the Models
Run the training pipeline script to aggregate the dataset, tune hyperparameters, select the best model, and save files to `models/` and `plots/`:
```bash
python main.py
```

### Step 2: Start the Web Server
Start the FastAPI server using Uvicorn:
```bash
python -m uvicorn app:app --port 8000
```
Open **`http://localhost:8000`** in your browser to access the dashboard.

---

## Model Evaluation Results

Due to the physical linear relationship between active power, voltage, and current intensity ($P = V \times I$), all models achieve near-perfect evaluation scores:

| Model | MAE (kW) | RMSE (kW) | R² Score | MAPE (%) |
|-------|:---:|:---:|:---:|:---:|
| 🏆 **Gradient Boosting** *(Selected Best)* | **0.0134** | **0.0186** | **99.96% (0.9996)** | **2.46%** |
| 🌲 **Linear Regression** | 0.0172 | 0.0226 | 99.94% (0.9994) | 2.94% |
| ⚡ **Random Forest** | 0.0181 | 0.0370 | 99.83% (0.9983) | 3.01% |

*Note: The FastAPI Predict page uses the best overall model (Gradient Boosting) to calculate power outputs. For transparency, it also displays a KaTeX equation and a step-by-step contribution breakdown using the highly interpretable weights of the Linear Regression model.*

---

## Web Dashboard Features

* **Overview View**: Showcases the best production model metrics and displays a **live SVG flowchart diagram** that dynamically highlights the best model and animates connection lines on metrics load.
* **Exploratory Data View**: Includes full summary statistics and displays time series, distribution, correlation heatmaps, and seasonal averages.
* **Performance View**: Summarizes model metrics side-by-side.
* **Predict View**: Includes a 7-parameter prediction form, a dynamic LaTeX formula viewer, and a step-by-step contribution breakdown table showing exactly how each variable influenced the forecast.

---

## Quick Start Run Commands

Here is the quick sequence of commands to get the project up and running locally:

```bash
# 1. Clone and enter the project folder
git clone https://github.com/sanjith-sanil/EnergyPredictionML.git
cd EnergyPredictionML

# 2. Create and activate a Python virtual environment
python -m venv .venv
.venv\Scripts\activate  # On macOS/Linux use: source .venv/bin/activate

# 3. Install required packages
pip install -r requirements.txt

# 4. Place household_power_consumption.csv into this folder, then run the ML pipeline
python main.py

# 5. Start the FastAPI backend server
python -m uvicorn app:app --port 8000
```
Once started, open your web browser and navigate to **`http://localhost:8000`** to view the application!
