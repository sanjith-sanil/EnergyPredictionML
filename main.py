# =============================================================================
# Power Consumption Prediction using Machine Learning and Smart Meter Data
# main.py — Full ML Pipeline
# =============================================================================

import os
import sys
import warnings

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import joblib

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# Ensure output directories exist
os.makedirs('plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# === 1. DATA LOADING & CLEANING ===
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  SECTION 1: DATA LOADING & CLEANING")
print("="*70)

# --- Load household power consumption ---
print("\n[1.1] Loading household power consumption data...")
power_df = pd.read_csv(
    'household_power_consumption.csv',
    sep=',',
    na_values=['?'],
    low_memory=False
)

# Parse datetime
power_df['Datetime'] = pd.to_datetime(
    power_df['Date'] + ' ' + power_df['Time'],
    format='%d/%m/%Y %H:%M:%S',
    dayfirst=True
)
power_df.set_index('Datetime', inplace=True)
power_df.drop(columns=['Date', 'Time'], inplace=True)

# Coerce all columns to numeric
for col in power_df.columns:
    power_df[col] = pd.to_numeric(power_df[col], errors='coerce')

print(f"  Raw power data shape: {power_df.shape}")
print(f"  Date range: {power_df.index.min()} to {power_df.index.max()}")
missing_count = power_df['Global_active_power'].isna().sum()
print(f"  Missing rows (Global_active_power): {missing_count:,}")

# Interpolate missing values (time-based)
power_df.interpolate(method='time', inplace=True)
power_df.dropna(inplace=True)
print(f"  After cleaning: {power_df.shape[0]:,} rows")

# --- Load weather data ---
print("\n[1.2] Loading weather data...")
weather_df = pd.read_csv('sceaux_weather_data.csv')
weather_df['time'] = pd.to_datetime(weather_df['time'])
weather_df.set_index('time', inplace=True)

# Drop near-empty columns
cols_to_drop = [c for c in ['snow', 'wpgt', 'tsun'] if c in weather_df.columns]
weather_df.drop(columns=cols_to_drop, inplace=True)
print(f"  Dropped sparse weather columns: {cols_to_drop}")

# Impute remaining missing weather values
for col in ['wdir', 'pres', 'wspd']:
    if col in weather_df.columns:
        weather_df[col] = weather_df[col].ffill()
        weather_df[col] = weather_df[col].fillna(weather_df[col].median())

print(f"  Weather data shape: {weather_df.shape}")
print(f"  Weather date range: {weather_df.index.min().date()} to {weather_df.index.max().date()}")

# ─────────────────────────────────────────────────────────────────────────────
# === 2. AGGREGATION & MERGING ===
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  SECTION 2: AGGREGATION & MERGING")
print("="*70)

# Resample power data to hourly mean
print("\n[2.1] Resampling power data to hourly averages...")
hourly_power = power_df.resample('H').mean()
print(f"  Hourly power data shape: {hourly_power.shape}")

# Merge on hourly datetime with daily weather forward-filled
print("[2.2] Merging power and weather datasets...")
weather_hourly = weather_df.resample('H').ffill()
merged_df = hourly_power.join(weather_hourly, how='inner')
merged_df.dropna(subset=['Global_active_power'], inplace=True)
print(f"  Merged dataset shape: {merged_df.shape}")
print(f"  Date range after merge: {merged_df.index.min().date()} to {merged_df.index.max().date()}")

# ─────────────────────────────────────────────────────────────────────────────
# === 3. FEATURE ENGINEERING ===
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  SECTION 3: FEATURE ENGINEERING")
print("="*70)

df = merged_df.copy()

# 3.1 Time features
print("\n[3.1] Creating time features...")
df['hour']        = df.index.hour
df['day']         = df.index.day
df['day_of_week'] = df.index.dayofweek          # 0=Monday
df['month']       = df.index.month

# Season: 0=Winter, 1=Spring, 2=Summer, 3=Autumn
def get_season(month):
    if month in [12, 1, 2]:  return 0
    elif month in [3, 4, 5]: return 1
    elif month in [6, 7, 8]: return 2
    else:                    return 3

df['season']  = df['month'].apply(get_season)

# 3.2 Drop Sub_metering_3 only (data leakage — keep Sub_metering_1 and Sub_metering_2 as requested)
print("[3.2] Dropping Sub_metering_3 (data leakage)...")
if 'Sub_metering_3' in df.columns:
    df.drop(columns=['Sub_metering_3'], inplace=True)

# 3.3 Lag features (key accuracy drivers)
print("[3.3] Creating lag features...")
df['lag_1'] = df['Global_active_power'].shift(1)
df['lag_2'] = df['Global_active_power'].shift(2)
df['lag_3'] = df['Global_active_power'].shift(3)
df['lag_24'] = df['Global_active_power'].shift(24)

# Drop rows with NaN from lag (first 24 rows)
df.dropna(inplace=True)
print(f"  Dataset after dropping NaN rows: {df.shape[0]} rows")

# 3.5 Rename ALL features to plain English labels (Layer 2 naming)
print("\n[3.5] Renaming features to plain English labels...")

RENAME_MAP = {
    'Global_active_power':  'Electricity Consumption (kW)',
    'Global_reactive_power':'Global_reactive_power',       # not used as feature
    'Voltage':              'Voltage (V)',
    'Global_intensity':     'Current Intensity (A)',
    'Sub_metering_1':       'Sub Metering 1 (Kitchen Wh)',
    'Sub_metering_2':       'Sub Metering 2 (Laundry Wh)',
    'tavg':                 'Temperature (°C)',
    'wspd':                 'Wind Speed (km/h)',
    'prcp':                 'Rainfall (mm)',
    'hour':                 'Hour of Day',
    'day':                  'Day of Month',
    'day_of_week':          'Day of Week',
    'month':                'Month',
    'season':               'Season',
    'lag_1':                "Consumption 1 Hour Ago (kW)",
    'lag_2':                "Consumption 2 Hours Ago (kW)",
    'lag_3':                "Consumption 3 Hours Ago (kW)",
    'lag_24':               "Consumption 24 Hours Ago (kW)",
}

df.rename(columns=RENAME_MAP, inplace=True)
print("  Rename complete.")

TARGET = 'Electricity Consumption (kW)'

# Exclude target and reactive power (not selected per spec — no plain English label assigned)
EXCLUDE = [TARGET, 'Global_reactive_power']
ALL_FEATURES = [c for c in df.columns if c not in EXCLUDE]
print(f"  Total features before selection: {len(ALL_FEATURES)}")
print(f"  Features: {ALL_FEATURES}")

# ─────────────────────────────────────────────────────────────────────────────
# === 4. FEATURE SELECTION ===
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  SECTION 4: FEATURE SELECTION (Using Exact User Selection)")
print("="*70)

# Define exact features requested by user
selected_features = [
    'Temperature (°C)',
    'Hour of Day',
    'Season',
    'Voltage (V)',
    'Consumption 1 Hour Ago (kW)',
    'Consumption 24 Hours Ago (kW)',
    'Current Intensity (A)',
]

print(f"  Selected exact {len(selected_features)} features as specified.")

# Compute Random Forest feature importance ranking on these features for downstream components
print("\n[4.1] Computing Random Forest feature importances...")
X_sel = df[selected_features].values
y_sel = df[TARGET].values

rf_selector = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_selector.fit(X_sel, y_sel)

importances = pd.Series(
    rf_selector.feature_importances_,
    index=selected_features
).sort_values(ascending=False)

print(f"  {'Feature':<45} {'Importance':>12}")
print(f"  {'-'*57}")
for feat in selected_features:
    imp = importances.get(feat, 0.0)
    print(f"  {feat:<45} {imp:>12.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# === 6. MODEL OPTIMIZATION (runs before step 5) ===
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  SECTION 6: MODEL OPTIMIZATION (Hyperparameter Tuning)")
print("="*70)

# Prepare train/test split for tuning (80/20, time-ordered)
X = df[selected_features].values
y = df[TARGET].values
dates = df.index

split_idx = int(len(X) * 0.80)
X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
y_train, y_test         = y[:split_idx], y[split_idx:]
dates_train, dates_test = dates[:split_idx], dates[split_idx:]

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

# Two separate CV objects:
# - tscv_tune: n_splits=3 for RandomizedSearchCV (larger folds on small dataset)
# - tscv_meta: n_splits=5 for OOF generation (as specified)
tscv_tune = TimeSeriesSplit(n_splits=3)
tscv_meta = TimeSeriesSplit(n_splits=5)

# --- Tune Random Forest ---
print("\n[6.1] Tuning RandomForestRegressor (n_iter=5)...")
rf_param_grid = {
    'n_estimators':    [200, 300, 500],
    'max_depth':       [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
    'max_features':    ['sqrt', 'log2', 0.5],
}
rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions=rf_param_grid,
    n_iter=5,
    cv=tscv_tune,
    scoring='r2',
    random_state=42,
    n_jobs=-1,
    verbose=0
)
rf_search.fit(X_train, y_train)
best_rf_params = rf_search.best_params_
print(f"  Best RF parameters: {best_rf_params}")
print(f"  Best RF CV R²: {rf_search.best_score_:.4f}")

# --- Tune Gradient Boosting ---
print("\n[6.2] Tuning GradientBoostingRegressor (n_iter=5)...")
gb_param_grid = {
    'n_estimators':      [200, 300, 500],
    'max_depth':         [3, 5, 7],
    'learning_rate':     [0.01, 0.05, 0.1],
    'subsample':         [0.7, 0.8, 1.0],
    'min_samples_split': [2, 5, 10],
}
gb_search = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_distributions=gb_param_grid,
    n_iter=5,
    cv=tscv_tune,
    scoring='r2',
    random_state=42,
    n_jobs=-1,
    verbose=0
)
gb_search.fit(X_train, y_train)
best_gb_params = gb_search.best_params_
print(f"  Best GB parameters: {best_gb_params}")
print(f"  Best GB CV R²: {gb_search.best_score_:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# === 5. BASE MODEL TRAINING & SELECTION ===
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  SECTION 5: BASE MODEL TRAINING")
print("="*70)

# Base models with best hyperparameters
base_models = {
    'Linear Regression':    LinearRegression(),
    'Random Forest':        RandomForestRegressor(random_state=42, n_jobs=-1, **best_rf_params),
    'Gradient Boosting':    GradientBoostingRegressor(random_state=42, **best_gb_params),
}

# Train base models on full training set
print("\n[5.1] Training base models on full training set...")
for name, model in base_models.items():
    model.fit(X_train, y_train)
    print(f"  Trained: {name}")

print("  Base models training complete.")

# ─────────────────────────────────────────────────────────────────────────────
# === 7. EVALUATION ===
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  SECTION 7: MODEL EVALUATION & SELECTION")
print("="*70)

def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def compute_metrics(y_true, y_pred, name):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape_val = mape(y_true, y_pred)
    return {'Model': name, 'MAE': mae, 'RMSE': rmse, 'R²': r2, 'MAPE (%)': mape_val}

all_test_preds = {}
all_metrics    = []

# Evaluate each base model
for name, model in base_models.items():
    preds = model.predict(X_test)
    all_test_preds[name] = preds
    all_metrics.append(compute_metrics(y_test, preds, name))

# Print comparison table
metrics_df = pd.DataFrame(all_metrics).set_index('Model')
print("\n  -- Electricity Consumption (kW) -- Model Performance --")
print(f"\n  {'Model':<25} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'MAPE (%)':>10}")
print(f"  {'-'*62}")
for model_name, row in metrics_df.iterrows():
    print(f"  {model_name:<25} {row['MAE']:>8.4f} {row['RMSE']:>8.4f} {row['R²']:>8.4f} {row['MAPE (%)']:>10.2f}")

# Select the best model programmatically based on R²
best_model_name = metrics_df['R²'].idxmax()
best_r2 = metrics_df.loc[best_model_name, 'R²']
best_model = base_models[best_model_name]
best_preds = all_test_preds[best_model_name]

print(f"\n  Selected Best Model: {best_model_name} (R²: {best_r2:.4f})")

# ─────────────────────────────────────────────────────────────────────────────
# === 8. SAVE ARTIFACTS ===
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  SECTION 8: SAVING ARTIFACTS")
print("="*70)

# 8.1 Build feature importance series (RF-based)
rf_importance = pd.Series(
    base_models['Random Forest'].feature_importances_,
    index=selected_features
).sort_values(ascending=False)

# 8.2 Build test results dataframe
test_results = pd.DataFrame({
    'Date':                        dates_test,
    'Electricity Consumption (kW)': y_test,
    'Linear Regression':            all_test_preds['Linear Regression'],
    'Random Forest':                all_test_preds['Random Forest'],
    'Gradient Boosting':            all_test_preds['Gradient Boosting'],
    'Best Model Predictions':       best_preds,
})

# 8.3 Save model artifacts
print("[8.1] Saving model artifacts to models/...")
joblib.dump(best_model,             'models/best_model.pkl')
joblib.dump(best_model,             'models/stacked_model.pkl') # legacy compatibility
joblib.dump(base_models,           'models/base_models.pkl')
joblib.dump(scaler,                'models/scaler.pkl')
joblib.dump(selected_features,     'models/feature_names.pkl')
joblib.dump(metrics_df.to_dict(),  'models/metrics.pkl')
joblib.dump(test_results,          'models/test_results.pkl')
joblib.dump(rf_importance,         'models/feature_importance.pkl')
joblib.dump(best_model_name,       'models/best_model_name.pkl')
print("  Model artifacts saved.")

# ─────────────────────────────────────────────────────────────────────────────
# === PLOTS ===
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8.2] Generating plots...")

# Plotting style
plt.rcParams.update({
    'font.family':  'DejaVu Sans',
    'font.size':     11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi':    120,
})
PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

# ── Plot 1: Actual vs Predicted ──
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(dates_test, y_test, label='Actual', color='#2E86AB', linewidth=1.5, alpha=0.9)
ax.plot(dates_test, best_preds, label=f'Best Model ({best_model_name})', color='#C73E1D',
        linewidth=1.5, linestyle='--', alpha=0.9)
ax.set_title('Actual vs Predicted — Electricity Consumption (kW)', fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Electricity Consumption (kW)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=30)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('plots/actual_vs_predicted.png', bbox_inches='tight')
plt.close(fig)
print("  Saved: plots/actual_vs_predicted.png")

# ── Plot 2: Residuals ──
residuals = y_test - best_preds
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(best_preds, residuals, alpha=0.5, color='#2E86AB', edgecolors='white', linewidths=0.3)
axes[0].axhline(0, color='#C73E1D', linestyle='--', linewidth=1.5)
axes[0].set_xlabel('Predicted Electricity Consumption (kW)')
axes[0].set_ylabel('Residual (kW)')
axes[0].set_title('Residuals vs Predicted Values')
axes[0].grid(True, alpha=0.3)

axes[1].hist(residuals, bins=30, color='#A23B72', edgecolor='white', linewidth=0.5)
axes[1].axvline(0, color='#C73E1D', linestyle='--', linewidth=1.5)
axes[1].set_xlabel('Residual (kW)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Residuals')
axes[1].grid(True, alpha=0.3)

fig.suptitle('Residual Analysis — Electricity Consumption (kW)', fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig('plots/residuals.png', bbox_inches='tight')
plt.close(fig)
print("  Saved: plots/residuals.png")

# ── Plot 3: Feature Importance ──
top_n = min(15, len(rf_importance))
top_imp = rf_importance.head(top_n)
fig, ax = plt.subplots(figsize=(10, 6))
colors_imp = plt.cm.viridis(np.linspace(0.2, 0.85, top_n))
bars = ax.barh(top_imp.index[::-1], top_imp.values[::-1], color=colors_imp)
ax.set_xlabel('Importance Score')
ax.set_title('Feature Importance — Random Forest\n(Plain English Labels)', fontweight='bold')
ax.grid(True, axis='x', alpha=0.3)
for bar, val in zip(bars, top_imp.values[::-1]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9)
fig.tight_layout()
fig.savefig('plots/feature_importance.png', bbox_inches='tight')
plt.close(fig)
print("  Saved: plots/feature_importance.png")

# ── Plot 4: Time Series (EDA) ──
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(merged_df.index, merged_df['Global_active_power'], color='#2E86AB', linewidth=1.5)
ax.set_title("Energy Consumption Over Time (Hourly Average)", fontweight='bold')
ax.set_xlabel("Date")
ax.set_ylabel("Global_active_power (kW)")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=30)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('plots/eda_time_series.png', bbox_inches='tight')
plt.close(fig)
print("  Saved: plots/eda_time_series.png")

# ── Plot 5: Distribution Analysis (EDA) ──
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(merged_df['Global_active_power'], kde=True, color='#A23B72', ax=ax, edgecolor='white')
ax.set_title("Distribution of Hourly Energy Consumption", fontweight='bold')
ax.set_xlabel("Global_active_power (kW)")
ax.set_ylabel("Frequency")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('plots/eda_distribution.png', bbox_inches='tight')
plt.close(fig)
print("  Saved: plots/eda_distribution.png")

# ── Plot 6: Correlation Heatmap (EDA) ──
# Use merged_df with original column names
eda_cols = [c for c in merged_df.columns if c != 'Global_reactive_power']
eda_subset = merged_df[eda_cols].dropna()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    eda_subset.corr(), annot=True, fmt='.2f', cmap='coolwarm',
    center=0, linewidths=0.5, ax=ax,
    annot_kws={'size': 9}, cbar_kws={'shrink': 0.8}
)
ax.set_title('Correlation Heatmap (Original Column Names)', fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9.5)
plt.yticks(rotation=0, fontsize=9.5)
fig.tight_layout()
fig.savefig('plots/correlation_heatmap.png', bbox_inches='tight')
plt.close(fig)
print("  Saved: plots/correlation_heatmap.png")

# ── Plot 7: Seasonal Trend Graph (EDA) ──
# Map season numeric codes to text labels
seasonal_df = merged_df.copy()
seasonal_df['month_num'] = seasonal_df.index.month
seasonal_df['Season Label'] = seasonal_df['month_num'].apply(
    lambda m: 'Winter' if m in [12, 1, 2] else 'Spring' if m in [3, 4, 5] else 'Summer' if m in [6, 7, 8] else 'Autumn'
)
seasonal_mean = seasonal_df.groupby('Season Label')['Global_active_power'].mean().reindex(['Winter', 'Spring', 'Summer', 'Autumn'])

fig, ax = plt.subplots(figsize=(8, 5))
colors_sea = ['#2E86AB', '#2ECC71', '#F18F01', '#E74C3C']
bars = ax.bar(seasonal_mean.index, seasonal_mean.values, color=colors_sea, edgecolor='white', linewidth=0.5, width=0.6)
ax.set_xlabel("Season")
ax.set_ylabel("Average Global_active_power (kW)")
ax.set_title("Average Energy Consumption by Season", fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{bar.get_height():.3f} kW', ha='center', va='bottom', fontsize=10, fontweight='bold')
fig.tight_layout()
fig.savefig('plots/eda_seasonal.png', bbox_inches='tight')
plt.close(fig)
print("  Saved: plots/eda_seasonal.png")

print("\n" + "="*70)
print("  PIPELINE COMPLETE")
print(f"  Best Model ({best_model_name}) -- R2: {best_r2:.4f} | MAE: {metrics_df.loc[best_model_name,'MAE']:.4f} | RMSE: {metrics_df.loc[best_model_name,'RMSE']:.4f}")
print("  Run: streamlit run app.py")
print("="*70 + "\n")
