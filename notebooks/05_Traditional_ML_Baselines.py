"""
Traditional Machine Learning Baselines for COVID-19 Ventilator Demand Forecasting

This script implements traditional ML models as baselines for comparison with deep learning approaches:
- ARIMA/SARIMAX Time Series Models
- Gradient Boosting Regressor
- Support Vector Regression (SVR)
- Linear Regression

Reference: Oyelakin et al. (2023) "Deep Learning-Based COVID-19 Hospitalization Forecasting"
           IEEE ICMLA 2023, Jacksonville, FL, USA
"""

# %% [markdown]
# # Traditional Machine Learning Baselines
# 
# This notebook implements traditional ML approaches for ventilator demand forecasting,
# serving as baselines for comparison with deep learning models.

# %%
# -----------------------------------
# Setup and Imports
# -----------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

# Time Series Models
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

# Machine Learning Models
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)

# %%
# -----------------------------------
# Data Loading
# -----------------------------------
# Define Source Data Path
source_data = Path("../data/")

# Load processed data
data = pd.read_csv(source_data / "processed" / "merged_nhs_covid_data.csv")
print("Data loaded successfully!")
print(f"Shape: {data.shape}")

# Convert date to datetime
data['date'] = pd.to_datetime(data['date'])

# Aggregate to England level
data = data.groupby('date').agg({
    'covidOccupiedMVBeds': 'sum',
    'cumAdmissions': 'sum',
    'hospitalCases': 'sum',
    'newAdmissions': 'sum',
    'new_confirmed': 'sum',
    'new_deceased': 'sum',
    'cumulative_confirmed': 'sum',
    'cumulative_deceased': 'sum',
    'population': 'sum',
}).reset_index()
data['areaName'] = 'England'

print(f"Aggregated data shape: {data.shape}")

# %%
# -----------------------------------
# Vax Index Calculation (Same as deep learning models)
# -----------------------------------
def calculate_vax_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Vax index based on vaccination rates and efficacy across age groups.
    
    This is a synthetic vaccination index used to capture vaccination effects
    on hospitalization rates.
    """
    total_population = 60_000_000
    number_of_age_groups = 5
    vaccine_efficacy_first_dose = [0.89, 0.427, 0.76, 0.854, 0.75]
    vaccine_efficacy_second_dose = [0.92, 0.86, 0.81, 0.85, 0.80]
    age_group_probabilities_icu = [0.01, 0.02, 0.05, 0.1, 0.15]
    monthly_vaccination_rate_increase = 0.05
    vaccination_start_date = pd.Timestamp('2021-01-18')
    
    population_per_age_group = total_population / number_of_age_groups
    vax_index_list = []
    monthly_vaccination_rate = 0.0
    
    for index, row in df.iterrows():
        if row['date'].day == 1 and row['date'] >= vaccination_start_date:
            monthly_vaccination_rate += monthly_vaccination_rate_increase
            monthly_vaccination_rate = min(monthly_vaccination_rate, 1.0)
        
        Si_sum = 0.0
        for i in range(number_of_age_groups):
            vaccinated_population = monthly_vaccination_rate * population_per_age_group
            aij = vaccinated_population / 2
            bij = vaccinated_population / 2
            cij = population_per_age_group - aij - bij
            S_double_prime_i = (vaccine_efficacy_second_dose[i] * bij +
                               vaccine_efficacy_first_dose[i] * aij)
            Si = aij + bij + cij - S_double_prime_i
            pi = age_group_probabilities_icu[i]
            Si_normalized = Si / population_per_age_group
            Si_sum += pi * Si_normalized
        
        vax_index = Si_sum
        vax_index_list.append(vax_index)
    
    df['Vax_index'] = vax_index_list
    return df

data = calculate_vax_index(data)
print("Vax_index calculated successfully!")

# %%
# -----------------------------------
# Feature Engineering for Traditional ML
# -----------------------------------
# Create time-based features
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['day_of_week'] = data['date'].dt.dayofweek

# Define key columns for feature engineering
selected_columns = [
    'covidOccupiedMVBeds',
    'hospitalCases',
    'newAdmissions',
    'new_confirmed',
    'Vax_index'
]

# Create lagged variables (7-day lag)
for column in selected_columns:
    data[f'{column}_lag7'] = data[column].shift(7)

# Create rolling averages (7-day rolling window)
for column in selected_columns:
    data[f'{column}_rolling7'] = data[column].rolling(window=7).mean()

print(f"Features created. New data shape: {data.shape}")
print("\nFeature columns:")
print(data.columns.tolist())

# %%
# -----------------------------------
# Data Visualization: Seasonal Decomposition
# -----------------------------------
# Prepare time series data for decomposition
ts_data = data[['date', 'covidOccupiedMVBeds']].set_index('date')
ts_data.dropna(inplace=True)

# Decompose with weekly seasonality (7 days)
decomposition = seasonal_decompose(ts_data, model='additive', period=7)

# Plot decomposition
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 12))

decomposition.observed.plot(ax=ax1, title='Observed', color='blue')
ax1.set_ylabel('MV Beds Occupied')

decomposition.trend.plot(ax=ax2, title='Trend', color='red')
ax2.set_ylabel('MV Beds Occupied')

decomposition.seasonal.plot(ax=ax3, title='Seasonal', color='green')
ax3.set_ylabel('MV Beds Occupied')

decomposition.resid.plot(ax=ax4, title='Residual', color='gray')
ax4.set_ylabel('MV Beds Occupied')

plt.tight_layout()
plt.savefig('../figures/seasonal_decomposition.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# -----------------------------------
# Correlation Analysis
# -----------------------------------
# Select features for correlation analysis
correlation_columns = [
    'covidOccupiedMVBeds',
    'hospitalCases', 
    'newAdmissions',
    'new_confirmed',
    'Vax_index',
    'covidOccupiedMVBeds_lag7',
    'hospitalCases_lag7'
]

# Compute correlation matrix
correlation_data = data[correlation_columns].dropna()
correlation_matrix = correlation_data.corr()

# Visualize correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, fmt='.2f')
plt.title('Correlation Matrix of Key Variables', fontsize=14)
plt.tight_layout()
plt.savefig('../figures/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# -----------------------------------
# Prepare Data for Modeling
# -----------------------------------
target = 'covidOccupiedMVBeds'
exogenous_variables = [
    'new_confirmed_lag7',
    'hospitalCases_lag7',
    'Vax_index_rolling7'
]

# Drop rows with NaN values (from lag/rolling features)
model_data = data.dropna(subset=[target] + exogenous_variables)

# Train-test split (80-20, preserving time order)
train_data, test_data = train_test_split(model_data, test_size=0.2, shuffle=False)

print(f"Training samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")

# %%
# -----------------------------------
# ARIMA/SARIMAX Model
# -----------------------------------
print("=" * 60)
print("SARIMAX MODEL TRAINING")
print("=" * 60)

# Hyperparameter tuning for SARIMAX
parameter_combinations = [
    (1, 1, 1),
    (1, 1, 2),
    (2, 1, 1),
    (2, 1, 2),
    (3, 1, 1),
    (3, 1, 2),
    (4, 1, 1),
    (4, 1, 2),
    (5, 1, 1),
    (5, 1, 2),
]

best_mae = np.inf
best_rmse = np.inf
best_order = None
best_model = None

print("\nHyperparameter Search:")
for params in parameter_combinations:
    try:
        model = SARIMAX(
            train_data[target], 
            exog=train_data[exogenous_variables], 
            order=params
        )
        model_fit = model.fit(disp=False)
        
        # Forecasting
        forecast = model_fit.get_forecast(
            steps=len(test_data), 
            exog=test_data[exogenous_variables]
        )
        forecast_values = forecast.predicted_mean
        
        # Calculate metrics
        mae = mean_absolute_error(test_data[target], forecast_values)
        rmse = np.sqrt(mean_squared_error(test_data[target], forecast_values))
        
        print(f"  Order {params}: MAE = {mae:.4f}, RMSE = {rmse:.4f}")
        
        if mae < best_mae and rmse < best_rmse:
            best_mae = mae
            best_rmse = rmse
            best_order = params
            best_model = model_fit
    except Exception as e:
        print(f"  Order {params}: Failed - {str(e)[:50]}")
        continue

print(f"\nBest SARIMAX Order: {best_order}")
print(f"Best MAE: {best_mae:.4f}")
print(f"Best RMSE: {best_rmse:.4f}")

# Generate final forecast with best model
sarimax_forecast = best_model.get_forecast(
    steps=len(test_data), 
    exog=test_data[exogenous_variables]
)
sarimax_predictions = sarimax_forecast.predicted_mean
sarimax_conf_int = sarimax_forecast.conf_int()

# Plot SARIMAX results
plt.figure(figsize=(14, 6))
plt.plot(train_data['date'], train_data[target], label='Training Data', color='blue')
plt.plot(test_data['date'], test_data[target], label='Actual Test Data', color='green')
plt.plot(test_data['date'], sarimax_predictions, label=f'SARIMAX{best_order} Forecast', color='red')
plt.fill_between(
    test_data['date'],
    sarimax_conf_int.iloc[:, 0],
    sarimax_conf_int.iloc[:, 1],
    color='pink',
    alpha=0.3,
    label='95% Confidence Interval'
)
plt.title(f'SARIMAX{best_order} Forecasting - MV Beds Occupied')
plt.xlabel('Date')
plt.ylabel('COVID-19 MV Beds Occupied')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../figures/sarimax_forecast.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# -----------------------------------
# Gradient Boosting Regressor
# -----------------------------------
print("=" * 60)
print("GRADIENT BOOSTING REGRESSOR TRAINING")
print("=" * 60)

# Prepare features for ML models
predictors = [col for col in model_data.columns if 'lag7' in col or 'rolling7' in col]
predictors.append('day_of_week')

# Ensure no NaN values
ml_data = model_data.dropna(subset=[target] + predictors)

# Split data
train_ml, test_ml = train_test_split(ml_data, test_size=0.2, shuffle=False)

# Train Gradient Boosting model
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb_model.fit(train_ml[predictors], train_ml[target])

# Predictions
gb_train_predictions = gb_model.predict(train_ml[predictors])
gb_test_predictions = gb_model.predict(test_ml[predictors])

# Calculate metrics
gb_train_mae = mean_absolute_error(train_ml[target], gb_train_predictions)
gb_test_mae = mean_absolute_error(test_ml[target], gb_test_predictions)
gb_train_rmse = np.sqrt(mean_squared_error(train_ml[target], gb_train_predictions))
gb_test_rmse = np.sqrt(mean_squared_error(test_ml[target], gb_test_predictions))

print(f"Train MAE: {gb_train_mae:.4f} | Test MAE: {gb_test_mae:.4f}")
print(f"Train RMSE: {gb_train_rmse:.4f} | Test RMSE: {gb_test_rmse:.4f}")

# Plot Gradient Boosting results
plt.figure(figsize=(14, 6))
plt.plot(train_ml['date'], train_ml[target], label='Training Data', color='blue')
plt.plot(test_ml['date'], test_ml[target], label='Actual Test Data', color='green')
plt.plot(test_ml['date'], gb_test_predictions, label='Gradient Boosting Forecast', color='orange')
plt.title('Gradient Boosting Forecasting - MV Beds Occupied')
plt.xlabel('Date')
plt.ylabel('COVID-19 MV Beds Occupied')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../figures/gradient_boosting_forecast.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature importance
feature_importance = pd.DataFrame({
    'feature': predictors,
    'importance': gb_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Feature Importance')
plt.title('Gradient Boosting - Feature Importance')
plt.tight_layout()
plt.savefig('../figures/gb_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# -----------------------------------
# Support Vector Regression (SVR)
# -----------------------------------
print("=" * 60)
print("SUPPORT VECTOR REGRESSION TRAINING")
print("=" * 60)

# Train SVR model
svr_model = SVR(kernel='rbf', C=1.0, gamma='scale')
svr_model.fit(train_ml[predictors], train_ml[target])

# Predictions
svr_train_predictions = svr_model.predict(train_ml[predictors])
svr_test_predictions = svr_model.predict(test_ml[predictors])

# Calculate metrics
svr_train_mae = mean_absolute_error(train_ml[target], svr_train_predictions)
svr_test_mae = mean_absolute_error(test_ml[target], svr_test_predictions)
svr_train_rmse = np.sqrt(mean_squared_error(train_ml[target], svr_train_predictions))
svr_test_rmse = np.sqrt(mean_squared_error(test_ml[target], svr_test_predictions))

print(f"Train MAE: {svr_train_mae:.4f} | Test MAE: {svr_test_mae:.4f}")
print(f"Train RMSE: {svr_train_rmse:.4f} | Test RMSE: {svr_test_rmse:.4f}")

# Plot SVR results
plt.figure(figsize=(14, 6))
plt.plot(train_ml['date'], train_ml[target], label='Training Data', color='blue')
plt.plot(test_ml['date'], test_ml[target], label='Actual Test Data', color='green')
plt.plot(test_ml['date'], svr_test_predictions, label='SVR Forecast', color='purple')
plt.title('Support Vector Regression Forecasting - MV Beds Occupied')
plt.xlabel('Date')
plt.ylabel('COVID-19 MV Beds Occupied')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../figures/svr_forecast.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# -----------------------------------
# Linear Regression (Simple Baseline)
# -----------------------------------
print("=" * 60)
print("LINEAR REGRESSION TRAINING")
print("=" * 60)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(train_ml[predictors], train_ml[target])

# Predictions
lr_train_predictions = lr_model.predict(train_ml[predictors])
lr_test_predictions = lr_model.predict(test_ml[predictors])

# Calculate metrics
lr_train_mae = mean_absolute_error(train_ml[target], lr_train_predictions)
lr_test_mae = mean_absolute_error(test_ml[target], lr_test_predictions)
lr_train_rmse = np.sqrt(mean_squared_error(train_ml[target], lr_train_predictions))
lr_test_rmse = np.sqrt(mean_squared_error(test_ml[target], lr_test_predictions))

print(f"Train MAE: {lr_train_mae:.4f} | Test MAE: {lr_test_mae:.4f}")
print(f"Train RMSE: {lr_train_rmse:.4f} | Test RMSE: {lr_test_rmse:.4f}")

# %%
# -----------------------------------
# Model Comparison
# -----------------------------------
print("=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)

# Compile all metrics
comparison_metrics = {
    'Model': ['SARIMAX', 'Gradient Boosting', 'SVR', 'Linear Regression'],
    'Test MAE': [best_mae, gb_test_mae, svr_test_mae, lr_test_mae],
    'Test RMSE': [best_rmse, gb_test_rmse, svr_test_rmse, lr_test_rmse]
}

metrics_df = pd.DataFrame(comparison_metrics)
metrics_df = metrics_df.sort_values('Test MAE')
print("\nTraditional ML Model Comparison:")
print(metrics_df.to_string(index=False))

# Save metrics to CSV
metrics_df.to_csv('../data/processed/traditional_ml_metrics.csv', index=False)

# Visualization of model comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# MAE comparison
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
axes[0].bar(metrics_df['Model'], metrics_df['Test MAE'], color=colors)
axes[0].set_ylabel('Mean Absolute Error (MAE)')
axes[0].set_title('Test MAE Comparison')
axes[0].tick_params(axis='x', rotation=45)

# RMSE comparison
axes[1].bar(metrics_df['Model'], metrics_df['Test RMSE'], color=colors)
axes[1].set_ylabel('Root Mean Squared Error (RMSE)')
axes[1].set_title('Test RMSE Comparison')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('../figures/traditional_ml_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# -----------------------------------
# Combined Forecast Visualization
# -----------------------------------
plt.figure(figsize=(16, 8))

plt.plot(test_ml['date'], test_ml[target], label='Actual', color='black', linewidth=2)
plt.plot(test_data['date'], sarimax_predictions.values, label=f'SARIMAX{best_order}', alpha=0.8)
plt.plot(test_ml['date'], gb_test_predictions, label='Gradient Boosting', alpha=0.8)
plt.plot(test_ml['date'], svr_test_predictions, label='SVR', alpha=0.8)
plt.plot(test_ml['date'], lr_test_predictions, label='Linear Regression', alpha=0.8)

plt.title('Traditional ML Models - Forecast Comparison', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('COVID-19 MV Beds Occupied', fontsize=12)
plt.legend(loc='upper right')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../figures/all_traditional_models_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("Traditional ML Baselines Analysis Complete!")
print("=" * 60)
