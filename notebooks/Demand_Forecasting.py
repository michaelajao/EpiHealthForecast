# %%
# COVID-19 Ventilator Demand Forecasting
# ======================================

# This notebook implements a forecasting model for COVID-19 ventilator bed occupancy
# using enhanced LSTM networks with comprehensive feature engineering.

# %%
# --- Setup and Imports ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from typing import Tuple

# For interactive visualizations
from itertools import cycle

# Deep learning imports
import torch
import random
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import holidays
from datetime import datetime, timedelta

# Configure paths
import sys
import os
sys.path.append(os.path.abspath('../'))

# From our custom modules
from src.utils import plotting_utils
from src.dl.dataloaders import TimeSeriesDataModule
from src.dl.multivariate_models import SingleStepRNNConfig, SingleStepRNNModel, Seq2SeqConfig, Seq2SeqModel, RNNConfig
from src.transforms.stationary_utils import check_seasonality, check_trend, check_heteroscedastisticity

# Set up progress bar for pandas operations
from tqdm.notebook import tqdm
tqdm.pandas()

# %%
# --- Configure Publication-Ready Visualizations ---
plt.style.use('seaborn-v0_8-paper')

# Set high resolution for publication-quality figures
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    
    # Typography configuration for scientific publications
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',  # Computer Modern math font
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'legend.title_fontsize': 14,
    'font.size': 14,
    
    # Optimize figure layout
    'figure.figsize': (8, 6),  # Approximates golden ratio
    'figure.constrained_layout.use': True,  # Better automatic spacing
    'axes.linewidth': 1.2,
    
    # Grid and line appearance
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': ':',
    'axes.axisbelow': True,
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
    'errorbar.capsize': 3,
    
    # Scientific color palette (colorblind-friendly)
    'axes.prop_cycle': plt.cycler('color', [
        '#0173B2', '#DE8F05', '#029E73', '#D55E00', 
        '#CC78BC', '#CA9161', '#FBAFE4', '#949494', 
        '#ECE133', '#56B4E9'
    ]),
    
    # Professional legend appearance
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'legend.edgecolor': 'k',
    'legend.facecolor': 'white',
    'legend.shadow': False,
    
    # Clean, precise axes and ticks
    'axes.spines.top': True,
    'axes.spines.right': True,
    'xtick.major.size': 6,
    'xtick.minor.size': 3,
    'ytick.major.size': 6,
    'ytick.minor.size': 3,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2
})

# %%
# --- Setting Seed for Reproducibility ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
pl.seed_everything(SEED, workers=True)

# Set PyTorch to use deterministic algorithms
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")

# %%
# --- Check for GPU Availability ---
if torch.cuda.is_available():
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
    device = torch.device('cuda')
    # Enable memory optimization
    torch.cuda.empty_cache()
else:
    print("GPU is not available, using CPU")
    device = torch.device('cpu')

# %%
# --- Load Data ---
source_data = Path("../data")

# Load the data
try:
    data = pd.read_csv(source_data / "processed" / "merged_nhs_covid_data.csv")
    print(f"Data loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns")
except FileNotFoundError:
    print("Data file not found. Please check the path.")
    
# Display the first few rows
data.head()

# %%
# --- Check Data Quality ---
# Check for missing values
missing_values = data.isnull().sum()
print("Missing values per column:")
print(missing_values[missing_values > 0])

# List all the unique values in the 'areaName' column
unique_areas = data['areaName'].unique()
print(f"Unique areas: {unique_areas}")
print(f"Number of unique areas: {len(unique_areas)}")

# %%
# --- Aggregate Data to England Level ---
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
    'openstreetmap_id': 'first',
    'latitude': 'first',
    'longitude': 'first'
}).reset_index()
data['areaName'] = 'England'

# Convert date to datetime
data['date'] = pd.to_datetime(data['date'])

# %%
# --- Stationarity and Seasonality Analysis ---
# Check for stationarity and seasonality in our target variable
target_series = data['covidOccupiedMVBeds']

# Check for trend
trend_result = check_trend(target_series, confidence=0.05)
print(f"Trend detected: {trend_result.trend}, Direction: {trend_result.direction}")

# Check for seasonality
seasonality_result = check_seasonality(target_series, max_lag=365)
print(f"Seasonality detected: {seasonality_result.seasonal}, Period: {seasonality_result.seasonal_periods}")

# Check for heteroscedasticity
hetero_result = check_heteroscedastisticity(target_series)
print(f"Heteroscedasticity detected: {hetero_result.heteroscedastic}")

# %%
# --- Synthetic Vaccination Index Creation ---
def calculate_vax_index(df):
    """
    Calculate a synthetic vaccination index that models the effect of vaccination on ICU admission risk.
    This is an epidemiological model that estimates population immunity over time.
    
    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing date and other COVID-19 related columns
    
    Returns:
    -------
    pd.DataFrame
        DataFrame with an additional 'Vax_index' column
    """
    # Constants
    total_population = 60_000_000  # Approximate England population
    number_of_age_groups = 5
    
    # Vaccine efficacy by age group (first dose)
    vaccine_efficacy_first_dose = [0.89, 0.427, 0.76, 0.854, 0.75]
    
    # Vaccine efficacy by age group (second dose)
    vaccine_efficacy_second_dose = [0.92, 0.86, 0.81, 0.85, 0.80]
    
    # ICU admission risk by age group
    age_group_probabilities_icu = [0.01, 0.02, 0.05, 0.1, 0.15]
    
    # Monthly increase in vaccination rate after vaccination starts
    monthly_vaccination_rate_increase = 0.05
    
    # Vaccination program start date
    vaccination_start_date = pd.Timestamp('2021-01-18')
    
    # Population per age group (equal distribution for simplicity)
    population_per_age_group = total_population / number_of_age_groups
    
    # Initialize Vax index list
    vax_index_list = []
    
    # Monthly vaccination rate (starting from 0)
    monthly_vaccination_rate = 0.0
    
    for index, row in df.iterrows():
        # Increment monthly vaccination rate on the first day of each month after start date
        if row['date'].day == 1 and row['date'] >= vaccination_start_date:
            monthly_vaccination_rate += monthly_vaccination_rate_increase
            # Ensure vaccination rate does not exceed 1
            monthly_vaccination_rate = min(monthly_vaccination_rate, 1.0)
            if index % 30 == 0:  # Reduce verbosity of output
                print(f"Updated monthly vaccination rate to {monthly_vaccination_rate:.2f} on {row['date'].strftime('%Y-%m-%d')}")
        
        Si_sum = 0.0
        
        for i in range(number_of_age_groups):
            # Vaccinated population for this age group
            vaccinated_population = monthly_vaccination_rate * population_per_age_group
            
            # Assume half received first dose and half received second dose
            aij = vaccinated_population / 2  # First dose
            bij = vaccinated_population / 2  # Second dose
            cij = population_per_age_group - aij - bij  # Unvaccinated
            
            # Calculate S''i based on vaccine efficacy (protected population)
            S_double_prime_i = (vaccine_efficacy_second_dose[i] * bij +
                               vaccine_efficacy_first_dose[i] * aij)
            
            # Calculate Si (effective susceptible population)
            Si = aij + bij + cij - S_double_prime_i  
            
            # Age-specific probability of ICU admission
            pi = age_group_probabilities_icu[i]
            
            # Normalize Si by total population in age group
            Si_normalized = Si / population_per_age_group
            
            # Weighted sum (vulnerability index)
            Si_sum += pi * Si_normalized
        
        # Vax index for the day (lower is better - less vulnerable population)
        vax_index = Si_sum
        vax_index_list.append(vax_index)
    
    # Add Vax index to the dataframe
    df['Vax_index'] = vax_index_list
    print("Calculated Vax_index for all dates.")
    return df

# Apply the vaccination index calculation
data = calculate_vax_index(data)

# %%
# --- Visualize Key Time Series Data (Matplotlib version) ---
fig, axs = plt.subplots(5, 1, figsize=(12, 15), sharex=True)

# Plot for New Hospital Admissions
axs[0].plot(data['date'], data['newAdmissions'], color='brown', linewidth=2)
axs[0].set_title('New Hospital Admissions')
axs[0].set_ylabel('Count')
axs[0].grid(True, alpha=0.3)

# Plot for Current Hospital Cases
axs[1].plot(data['date'], data['hospitalCases'], color='green', linewidth=2)
axs[1].set_title('Current Hospital Cases')
axs[1].set_ylabel('Count')
axs[1].grid(True, alpha=0.3)

# Plot for Mechanical Ventilators
axs[2].plot(data['date'], data['covidOccupiedMVBeds'], color='blue', linewidth=2)
axs[2].set_title('Mechanical Ventilator Bed Usage')
axs[2].set_ylabel('Count')
axs[2].grid(True, alpha=0.3)

# Plot for New Cases
axs[3].plot(data['date'], data['new_confirmed'], color='orange', linewidth=2)
axs[3].set_title('New COVID-19 Cases')
axs[3].set_ylabel('Count')
axs[3].grid(True, alpha=0.3)

# Plot for Vax Index
axs[4].plot(data['date'], data['Vax_index'], color='purple', linewidth=2)
axs[4].set_title('Vax Index')
axs[4].set_ylabel('Index Value')
axs[4].grid(True, alpha=0.3)

# Format x-axis dates
plt.xticks(rotation=45)
plt.xlabel('Date')
fig.tight_layout()
fig.suptitle('COVID-19 Data Visualization for England', fontsize=16, y=1.01)

# Save the figure
plt.savefig("../figures/covid_data_visualization.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
selected_columns = ['date', 'newAdmissions', 'hospitalCases', 'covidOccupiedMVBeds', 'new_confirmed', 'Vax_index']
# Aggregate data for England
england_data = data[selected_columns].groupby("date").sum(numeric_only=True).reset_index()

# Set Date as index for time series analysis
england_data.set_index('date', inplace=True)

# --- Correlation Heatmap ---
plt.figure(figsize=(10, 6))
sns.heatmap(england_data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap for ICU Bed Demand Forecasting Variables")
plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.savefig("../figures/correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller

# --- Seasonal Decomposition Analysis ---
if 'covidOccupiedMVBeds' in england_data.columns:
    decompose_result = seasonal_decompose(england_data['covidOccupiedMVBeds'], model='additive', period=30)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    decompose_result.observed.plot(ax=ax1, title='Observed (COVID-19 Occupied MV Beds)', color='blue')
    ax1.set_ylabel('Observed')

    decompose_result.trend.plot(ax=ax2, title='Trend', color='red')
    ax2.set_ylabel('Trend')

    decompose_result.seasonal.plot(ax=ax3, title='Seasonality', color='green')
    ax3.set_ylabel('Seasonality')

    decompose_result.resid.plot(ax=ax4, title='Residuals', color='grey')
    ax4.set_ylabel('Residuals')
    ax4.set_xlabel('Date')

    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.tight_layout()
    plt.savefig("../figures/seasonal_decomposition.png", dpi=600, bbox_inches='tight')
    plt.show()
else:
    print("Skipping decomposition analysis: 'covidOccupiedMVBeds' not found in dataset.")

# --- Time-Lagged Correlation Analysis ---
def plot_lag_correlation(df, feature, target, lags=30):
    if feature not in df.columns or target not in df.columns:
        print(f"Skipping lag correlation: Missing {feature} or {target} in dataset.")
        return
    correlation_values = [df[target].corr(df[feature].shift(lag)) for lag in range(lags + 1)]
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(lags + 1), correlation_values, marker='o')
    plt.title(f"Time-Lagged Correlation: {feature} → {target}")
    plt.xlabel('Lag (days)')
    plt.ylabel('Correlation')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.xticks(range(0, lags + 1, 5))
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.tight_layout()
    plt.savefig(f"../figures/lag_corr_{feature}_{target}.png", dpi=600, bbox_inches='tight')
    plt.show()

# Map your dataset columns to the analysis
predictors = ['newAdmissions', 'hospitalCases', 'new_confirmed', 'Vax_index']
target = 'covidOccupiedMVBeds'

for predictor in predictors:
    plot_lag_correlation(england_data, predictor, target, lags=30)

# %%
# --- Feature Engineering ---
# Calculate rolling statistics for key metrics
window_size_7 = 7   # Weekly rolling window
window_size_14 = 14  # Bi-weekly rolling window

# List of columns for which to compute rolling statistics
columns_to_compute = [
    'covidOccupiedMVBeds', 
    'hospitalCases', 
    'newAdmissions', 
    'Vax_index', 
    'new_confirmed'
]

# Compute rolling statistics for each column
for column in columns_to_compute:
    data[f'{column}_rolling_mean_7'] = data[column].rolling(window=window_size_7).mean()
    data[f'{column}_rolling_std_7'] = data[column].rolling(window=window_size_7).std()
    data[f'{column}_rolling_mean_14'] = data[column].rolling(window=window_size_14).mean()
    data[f'{column}_rolling_std_14'] = data[column].rolling(window=window_size_14).std()

# %%
def engineer_features(df):
    """
    Engineer comprehensive features for ventilator demand forecasting.
    
    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing COVID-19 data
        
    Returns:
    -------
    pd.DataFrame
        DataFrame with additional engineered features
    """
    # Create a copy to avoid modifying the original DataFrame
    data = df.copy()
    
    #------------------
    # LOCKDOWN FEATURES
    #------------------
    # Define lockdown periods for England
    lockdown_dates = {
        'Lockdown 1': {'start': '2020-03-23', 'end': '2020-07-04'},
        'Lockdown 2': {'start': '2020-11-05', 'end': '2020-12-02'},
        'Lockdown 3': {'start': '2021-01-06', 'end': '2021-04-12'}
    }
    
    # Create lockdown indicators
    data['in_lockdown'] = 0
    data['days_since_lockdown_start'] = np.nan
    data['days_until_lockdown_end'] = np.nan
    
    # Create features for each lockdown period
    for lockdown_name, period in lockdown_dates.items():
        start_date = pd.to_datetime(period['start'])
        end_date = pd.to_datetime(period['end'])
        
        # Create binary indicator for this specific lockdown
        lockdown_col = f'in_{lockdown_name.lower().replace(" ", "_")}'
        data[lockdown_col] = ((data['date'] >= start_date) & (data['date'] <= end_date)).astype(int)
        
        # Update the general lockdown indicator
        data.loc[(data['date'] >= start_date) & (data['date'] <= end_date), 'in_lockdown'] = 1
        
        # Calculate days since lockdown start (for dates within and after the lockdown)
        mask_since = data['date'] >= start_date
        data.loc[mask_since, f'days_since_{lockdown_name.lower().replace(" ", "_")}_start'] = (
            (data.loc[mask_since, 'date'] - start_date).dt.days
        )
        
        # Calculate days until lockdown end (for dates within the lockdown)
        mask_until = (data['date'] >= start_date) & (data['date'] <= end_date)
        data.loc[mask_until, f'days_until_{lockdown_name.lower().replace(" ", "_")}_end'] = (
            (end_date - data.loc[mask_until, 'date']).dt.days
        )
        
        # Fill NaN values with -1 for dates before the lockdown period
        data[f'days_since_{lockdown_name.lower().replace(" ", "_")}_start'] = data[f'days_since_{lockdown_name.lower().replace(" ", "_")}_start'].fillna(-1)
        data[f'days_until_{lockdown_name.lower().replace(" ", "_")}_end'] = data[f'days_until_{lockdown_name.lower().replace(" ", "_")}_end'].fillna(-1)
    
    # Calculate days since/until any lockdown
    for i, row in data.iterrows():
        current_date = row['date']
        
        # Find the closest lockdown start date that's in the past
        past_starts = [(pd.to_datetime(period['start']), name) 
                       for name, period in lockdown_dates.items() 
                       if pd.to_datetime(period['start']) <= current_date]
        
        if past_starts:
            closest_past_start = max(past_starts, key=lambda x: x[0])
            data.at[i, 'days_since_lockdown_start'] = (current_date - closest_past_start[0]).days
        else:
            data.at[i, 'days_since_lockdown_start'] = -1
            
        # Find the closest lockdown end date that's in the future
        future_ends = [(pd.to_datetime(period['end']), name) 
                       for name, period in lockdown_dates.items() 
                       if pd.to_datetime(period['start']) <= current_date <= pd.to_datetime(period['end'])]
        
        if future_ends:
            closest_future_end = min(future_ends, key=lambda x: x[0])
            data.at[i, 'days_until_lockdown_end'] = (closest_future_end[0] - current_date).days
        else:
            data.at[i, 'days_until_lockdown_end'] = -1
    
    # Create a feature for time since last lockdown ended
    data['days_since_last_lockdown'] = -1
    for i, row in data.iterrows():
        current_date = row['date']
        
        # Find past lockdown end dates
        past_ends = [(pd.to_datetime(period['end']), name) 
                    for name, period in lockdown_dates.items() 
                    if pd.to_datetime(period['end']) < current_date]
        
        if past_ends and data.at[i, 'in_lockdown'] == 0:
            most_recent_end = max(past_ends, key=lambda x: x[0])
            data.at[i, 'days_since_last_lockdown'] = (current_date - most_recent_end[0]).days
    
    #-------------------------
    # RATE OF CHANGE FEATURES
    #-------------------------
    # 1. Rate of change (daily) for hospital cases
    data['hospitalCases_daily_change'] = data['hospitalCases'].diff()
    data['hospitalCases_pct_change'] = data['hospitalCases'].pct_change() * 100
    
    # 2. Rate of change (daily) for new admissions
    data['newAdmissions_daily_change'] = data['newAdmissions'].diff()
    data['newAdmissions_pct_change'] = data['newAdmissions'].pct_change() * 100
    
    # 3. Rate of change in ventilator usage
    data['vent_daily_change'] = data['covidOccupiedMVBeds'].diff()
    data['vent_pct_change'] = data['covidOccupiedMVBeds'].pct_change() * 100
    
    # 4. Rate of change in confirmed cases
    if 'new_confirmed' in data.columns:
        data['confirmed_daily_change'] = data['new_confirmed'].diff()
        data['confirmed_pct_change'] = data['new_confirmed'].pct_change() * 100
    
    #-------------------
    # MOMENTUM FEATURES
    #-------------------
    # Multi-day momentum features (change over 3 and 7 days)
    for col in ['hospitalCases', 'newAdmissions', 'covidOccupiedMVBeds']:
        data[f'{col}_3day_momentum'] = data[col].diff(3)
        data[f'{col}_7day_momentum'] = data[col].diff(7)
    
    if 'new_confirmed' in data.columns:
        data['new_confirmed_3day_momentum'] = data['new_confirmed'].diff(3)
        data['new_confirmed_7day_momentum'] = data['new_confirmed'].diff(7)
    
    #---------------
    # RATIO FEATURES
    #---------------
    # 1. Percentage of hospital cases requiring ventilation
    data['pct_cases_ventilated'] = (data['covidOccupiedMVBeds'] / data['hospitalCases']).clip(0, 1) * 100
    
    # 2. Admission to hospital ratio
    data['admission_to_hospital_ratio'] = (data['newAdmissions'] / data['hospitalCases']).clip(0, 10)
    
    # 3. Ventilator to hospital ratio
    data['vent_to_hospital_ratio'] = (data['covidOccupiedMVBeds'] / data['hospitalCases']).clip(0, 1)
    
    # 4. Admission to ventilator ratio (proxy for severity)
    data['admission_to_vent_ratio'] = (data['newAdmissions'] / data['covidOccupiedMVBeds']).clip(0, 10)
    
    #---------------------
    # PEAK-RELATED FEATURES
    #---------------------
    # Days since most recent peak in key metrics
    def days_since_peak(series):
        result = np.zeros(len(series))
        current_max = series.iloc[0]
        current_max_idx = 0
        
        for i in range(1, len(series)):
            if series.iloc[i] > current_max:
                current_max = series.iloc[i]
                current_max_idx = i
            
            result[i] = i - current_max_idx
        
        return result
    
    data['days_since_vent_peak'] = days_since_peak(data['covidOccupiedMVBeds'])
    data['days_since_hospital_peak'] = days_since_peak(data['hospitalCases'])
    data['days_since_admissions_peak'] = days_since_peak(data['newAdmissions'])
    
    #----------------------
    # ACCELERATION FEATURES
    #----------------------
    # Change in the rate of change (acceleration)
    data['hospitalCases_acceleration'] = data['hospitalCases_daily_change'].diff()
    data['vent_acceleration'] = data['vent_daily_change'].diff()
    data['admissions_acceleration'] = data['newAdmissions_daily_change'].diff()
    
    #-------------------
    # TREND RATIO FEATURES
    #-------------------
    # Moving average ratios (Short-term vs long-term trends)
    if all(col in data.columns for col in ['hospitalCases_rolling_mean_7', 'hospitalCases_rolling_mean_14']):
        data['hospital_trend_ratio'] = (data['hospitalCases_rolling_mean_7'] / data['hospitalCases_rolling_mean_14']).fillna(1)
    
    if all(col in data.columns for col in ['covidOccupiedMVBeds_rolling_mean_7', 'covidOccupiedMVBeds_rolling_mean_14']):
        data['vent_trend_ratio'] = (data['covidOccupiedMVBeds_rolling_mean_7'] / data['covidOccupiedMVBeds_rolling_mean_14']).fillna(1)
    
    if all(col in data.columns for col in ['newAdmissions_rolling_mean_7', 'newAdmissions_rolling_mean_14']):
        data['admissions_trend_ratio'] = (data['newAdmissions_rolling_mean_7'] / data['newAdmissions_rolling_mean_14']).fillna(1)
    
    #-------------
    # LAG FEATURES
    #-------------
    # Lag features for key metrics (t-1, t-7, t-14)
    for col in ['covidOccupiedMVBeds', 'hospitalCases', 'newAdmissions']:
        data[f'{col}_lag_1'] = data[col].shift(1)
        data[f'{col}_lag_7'] = data[col].shift(7)
        data[f'{col}_lag_14'] = data[col].shift(14)
    
    if 'new_confirmed' in data.columns:
        data['new_confirmed_lag_1'] = data['new_confirmed'].shift(1)
        data['new_confirmed_lag_7'] = data['new_confirmed'].shift(7)
    
    #---------------
    # WAVE INDICATORS
    #---------------
    # COVID wave indicator using threshold approach
    def identify_waves(series, threshold_multiplier=1.5):
        # Use rolling mean to smooth the data
        smooth = series.rolling(window=14, min_periods=1).mean()
        # Calculate the overall mean
        overall_mean = smooth.mean()
        # Mark as wave when above threshold
        return (smooth > (overall_mean * threshold_multiplier)).astype(int)
    
    data['covid_wave'] = identify_waves(data['covidOccupiedMVBeds'])
    
    # Alternative wave indicator using rate of change
    data['wave_momentum'] = (data['covidOccupiedMVBeds_rolling_mean_7'].diff(7) > 0).astype(int)
    
    #-------------------
    # LOCKDOWN INTERACTION
    #-------------------
    # Interaction features between lockdown and other metrics
    data['lockdown_hospital_interaction'] = data['in_lockdown'] * data['hospitalCases']
    data['lockdown_admission_interaction'] = data['in_lockdown'] * data['newAdmissions']
    data['lockdown_vent_interaction'] = data['in_lockdown'] * data['covidOccupiedMVBeds']
    
    # Lockdown effectiveness metrics (change in key metrics during lockdown periods)
    for col in ['hospitalCases', 'newAdmissions', 'covidOccupiedMVBeds']:
        data[f'{col}_lockdown_effect'] = data[col].diff(7) * data['in_lockdown']
    
    # Fill NaN values with appropriate defaults
    # For percentage changes, use 0
    pct_cols = [col for col in data.columns if 'pct_' in col or '_pct' in col]
    data[pct_cols] = data[pct_cols].fillna(0)
    
    # For ratio columns, use forward fill then backfill
    ratio_cols = [col for col in data.columns if 'ratio' in col]
    data[ratio_cols] = data[ratio_cols].fillna(method='ffill').fillna(method='bfill').fillna(1)
    
    # For other columns with NaNs, use 0
    data = data.fillna(0)
    
    return data

# Apply enhanced feature engineering
enhanced_data = engineer_features(data)
print(f"Original data columns: {len(data.columns)}")
print(f"Enhanced data columns: {len(enhanced_data.columns)}")
print(f"New features added: {len(enhanced_data.columns) - len(data.columns)}")

# %%
# List all the selected columns (both features and target)
print("Selected columns:")
print(enhanced_data.columns.tolist())

# %%
# Drop redundant columns as described
drop_columns = [
    # Redundant location data
    "openstreetmap_id", "latitude", "longitude",
    # Redundant raw COVID metrics (keeping rolling averages instead)
    "hospitalCases", 
    # Cumulative metrics that may cause leakage
    "cumAdmissions", "cumulative_confirmed", "population",
    # Raw new cases vs. rolling mean (keep rolling mean)
    "new_confirmed",
    # Categorical variable needing encoding
    "areaName",
    # Overlapping lockdown indicators (drop phase-specific variables)
    "in_lockdown_1", "in_lockdown_2", "in_lockdown_3",
    "days_since_lockdown_1_start", "days_until_lockdown_1_end",
    "days_since_lockdown_2_start", "days_until_lockdown_2_end",
    "days_since_lockdown_3_start", "days_until_lockdown_3_end"
]

enhanced_data = enhanced_data.drop(columns=[col for col in drop_columns if col in enhanced_data.columns])
print("Columns remaining after dropping redundant features:")
print(enhanced_data.columns.tolist())

# %%
# --- Train / Validation / Test Split ---
# Find the minimum and maximum dates
min_date = enhanced_data['date'].min()
max_date = enhanced_data['date'].max()

print("Minimum Date:", min_date)
print("Maximum Date:", max_date)

# Calculate the date ranges for train, val, and test sets
date_range = max_date - min_date
train_end = min_date + pd.Timedelta(days=int(date_range.days * 0.75))
val_end = train_end + pd.Timedelta(days=int(date_range.days * 0.10))

# Split the data into train, validation, and test sets based on the date ranges
train = enhanced_data[enhanced_data['date'] < train_end]
val = enhanced_data[(enhanced_data['date'] >= train_end) & (enhanced_data['date'] < val_end)]
test = enhanced_data[enhanced_data['date'] >= val_end]

# Calculate the percentage of dates in each dataset
total_samples = len(enhanced_data)
train_percentage = len(train) / total_samples * 100
val_percentage = len(val) / total_samples * 100
test_percentage = len(test) / total_samples * 100

print(f"# of Training samples: {len(train)} | # of Validation samples: {len(val)} | # of Test samples: {len(test)}")
print(f"Percentage of Dates in Train: {train_percentage:.2f}% | Validation: {val_percentage:.2f}% | Test: {test_percentage:.2f}%")
print(f"Max Date in Train: {train.date.max()} | Min Date in Validation: {val.date.min()} | Min Date in Test: {test.date.min()}")

# %%
# --- Visualize the data split ---
plt.figure(figsize=(15, 6))
plt.plot(train['date'], train['covidOccupiedMVBeds'], 'b-', label='Training Set')
plt.plot(val['date'], val['covidOccupiedMVBeds'], 'g-', label='Validation Set')
plt.plot(test['date'], test['covidOccupiedMVBeds'], 'r-', label='Test Set')
plt.axvline(x=train_end, color='gray', linestyle='--', label='Train/Val Split')
plt.axvline(x=val_end, color='black', linestyle='--', label='Val/Test Split')
plt.title('COVID-19 Ventilator Bed Occupancy Data Split')
plt.xlabel('Date')
plt.ylabel('Number of Occupied Ventilator Beds')
plt.legend()
plt.tight_layout()
plt.savefig('../figures/data_split.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# --- Feature Selection and Data Preparation ---
# Convert date to datetime and set as index
train.set_index("date", inplace=True)
val.set_index("date", inplace=True)
test.set_index("date", inplace=True)

# Concatenate the DataFrames for reference
sample_df = pd.concat([train, val, test])

# Convert feature columns to float32 for efficiency and compatibility with deep learning
for col in sample_df.columns:
    if col != "type":
        sample_df[col] = sample_df[col].astype("float32")

print(f"Combined dataframe shape: {sample_df.shape}")

# %%
# --- Use feature importance from a baseline model to select relevant features ---
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import numpy as np

# Define the target variable
target = "covidOccupiedMVBeds"

# Prepare X and y for feature selection
X_train = train.drop(target, axis=1)
y_train = train[target]

# Clean the data: replace inf, -inf with NaN, then fill NaN with column means
X_train_cleaned = X_train.replace([np.inf, -np.inf], np.nan)
column_means = X_train_cleaned.mean()
X_train_cleaned = X_train_cleaned.fillna(column_means)

# Check if there are still any problematic values
print("Checking for remaining inf or NaN values:")
print(f"  Inf values: {np.any(np.isinf(X_train_cleaned.values))}")
print(f"  NaN values: {np.any(np.isnan(X_train_cleaned.values))}")

# Convert to float64 to handle potential precision issues
X_train_cleaned = X_train_cleaned.astype('float64')

# Track time for feature selection
start_time = time.time()
print("Starting feature selection...")

# Use correlation as a simple feature selection method
correlations = X_train_cleaned.corrwith(y_train).abs().sort_values(ascending=False)
top_corr_features = correlations.head(30).index.tolist()

# Additionally, try Random Forest with robust error handling
try:
    rf = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf.fit(X_train_cleaned, y_train)
    
    # Get feature importances
    importances = rf.feature_importances_
    feature_importances = pd.DataFrame({'feature': X_train_cleaned.columns, 'importance': importances})
    feature_importances = feature_importances.sort_values('importance', ascending=False)
    
    # Select top features from RF
    top_rf_features = feature_importances.head(30)['feature'].tolist()
    
    # Combine both selection methods
    top_features = list(set(top_corr_features + top_rf_features))
    selection_method = "Combined correlation and Random Forest"
except Exception as e:
    print(f"Random Forest feature selection failed with error: {e}")
    print("Falling back to correlation-based feature selection only")
    top_features = top_corr_features
    feature_importances = pd.DataFrame({'feature': correlations.index, 'importance': correlations.values})
    selection_method = "Correlation only"

print(f"Feature selection ({selection_method}) completed in {time.time() - start_time:.2f} seconds")
print(f"Selected {len(top_features)} most important features")

# %%
# --- Visualize feature importances ---
plt.figure(figsize=(12, 10))
sns.barplot(x='importance', y='feature', data=feature_importances.head(20))
plt.title(f'Top 20 Most Important Features ({selection_method})')
plt.tight_layout()
plt.savefig('../figures/feature_importance_selection.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# --- Define features based on our selection and domain knowledge ---
# Base features from key indicators
base_features = [
    'covidOccupiedMVBeds_lag_1',  # Previous day's ventilator beds
    'covidOccupiedMVBeds_lag_7',  # One week ago ventilator beds
    'covidOccupiedMVBeds_rolling_mean_7',  # Weekly average
    'covidOccupiedMVBeds_rolling_std_7',   # Weekly standard deviation
    'covidOccupiedMVBeds_rolling_mean_14',  # Bi-weekly average
    'covidOccupiedMVBeds_rolling_std_14',   # Bi-weekly standard deviation
    'hospitalCases',               # Current hospital cases
    'newAdmissions',               # New hospital admissions
    'new_confirmed',               # New COVID cases
    'Vax_index'                    # Vaccination effect
]

# Add selected features from the Random Forest
# Include top lockdown, ratio and momentum features from our feature importance ranking
selected_features = base_features.copy()
for feature in top_features:
    if (feature not in selected_features and 
        ('lockdown' in feature or 
         'ratio' in feature or 
         'momentum' in feature or 
         'acceleration' in feature or
         'wave' in feature or
         'peak' in feature)):
        selected_features.append(feature)

# Keep only features that exist in the dataframe
final_features = [f for f in selected_features if f in sample_df.columns]
print(f"Final feature count: {len(final_features)}")

# Select the features and the target variable for our model
selected_df = sample_df[final_features + [target]]

# %%
# --- Display correlation between features and target ---
correlation_with_target = selected_df.corr()[target].sort_values(ascending=False)
print("Top correlations with ventilator bed occupancy:")
print(correlation_with_target.head(10))
print("\nBottom correlations with ventilator bed occupancy:")
print(correlation_with_target.tail(5))

# %%
# --- Ensure the target column is the last column (important for the model) ---
cols = list(selected_df.columns)
if target in cols:
    cols.remove(target)
    selected_df = selected_df[cols + [target]]

# Display info about the selected dataframe
print("Selected dataframe info:")
selected_df.info()

# %%
# --- Model Development: Single-Step LSTM ---
# Create a TimeSeriesDataModule for the model
window_size = 14  # Use two weeks of data to predict
horizon = 1       # Predict one day ahead

datamodule = TimeSeriesDataModule(
    data=selected_df,
    n_val=val.shape[0],
    n_test=test.shape[0],
    window=window_size, 
    horizon=horizon,  
    normalize="global",  # Normalize across all data
    batch_size=32,
)
datamodule.setup()

# Get the actual input size from the datamodule
X_sample, _ = next(iter(datamodule.train_dataloader()))
actual_input_size = X_sample.shape[2]
print(f"Actual input size from dataloader: {actual_input_size}")

# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- Define model configurations ---
lstm_config = SingleStepRNNConfig(
    rnn_type="LSTM",  # Use LSTM instead of vanilla RNN
    input_size=actual_input_size,
    hidden_size=32,   # Increased from 64
    num_layers=5,      # Increased from 2 for better generalization
    bidirectional=True,  # Use bidirectional LSTM
    learning_rate=5e-4   # Decreased to avoid overshooting
)

# Create the LSTM model
lstm_model = SingleStepRNNModel(lstm_config)
lstm_model.to(device)

# Check total number of parameters
total_params = sum(p.numel() for p in lstm_model.parameters())
print(f"Total number of parameters: {total_params:,}")

# %%
# --- Model Training with Better Configurations ---
# Define callbacks for early stopping and model checkpointing
early_stopping = EarlyStopping(
    monitor="valid_loss",
    patience=10,
    verbose=True,
    mode="min"
)

model_checkpoint = ModelCheckpoint(
    monitor="valid_loss",
    dirpath="./checkpoints/",
    filename="covid-lstm-{epoch:02d}-{valid_loss:.4f}",
    save_top_k=1,
    mode="min"
)

# Train the model with improved settings
trainer = pl.Trainer(
    min_epochs=10,
    max_epochs=100,
    callbacks=[early_stopping, model_checkpoint],
    gradient_clip_val=0.5,  # Prevent exploding gradients
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1,
    precision='16-mixed' if torch.cuda.is_available() else '32',  # Use mixed precision if GPU is available
    log_every_n_steps=10
)

# Start timing the training
train_start_time = time.time()
print("Starting model training...")

# Train the model
trainer.fit(lstm_model, datamodule)

# Calculate training time
train_time = time.time() - train_start_time
print(f"Training completed in {train_time:.2f} seconds ({train_time/60:.2f} minutes)")

# %%
# --- Clean up any artifacts created during training ---
import shutil
if os.path.exists("lightning_logs"):
    shutil.rmtree("lightning_logs")

# %%
# --- Model Evaluation ---
# Define utility functions for metrics
def mase(actual, predicted, insample_actual):
    """
    Calculate the Mean Absolute Scaled Error (MASE).
    
    Args:
        actual (np.array): Actual values.
        predicted (np.array): Predicted values.
        insample_actual (np.array): In-sample actual values for scaling.
    
    Returns:
        float: The MASE value.
    """
    mae_insample = np.mean(np.abs(np.diff(insample_actual)))
    if mae_insample == 0:
        return np.nan
    mae_outsample = np.mean(np.abs(actual - predicted))
    return mae_outsample / mae_insample

def forecast_bias(actual, predicted):
    """
    Calculate the forecast bias.
    
    Args:
        actual (np.array): Actual values.
        predicted (np.array): Predicted values.
    
    Returns:
        float: The forecast bias.
    """
    return np.mean(predicted - actual)

# Evaluate the model and calculate metrics
metric_record = []

# Get predictions for test set
pred = trainer.predict(lstm_model, datamodule.test_dataloader())
pred = torch.cat(pred).squeeze().detach().numpy()

# Denormalize predictions using datamodule's statistics
pred = pred * datamodule.train.std + datamodule.train.mean

# Get actual values from test set
actuals = test[target].values

# Calculate metrics
metrics = {
    "Algorithm": lstm_config.rnn_type,
    "MAE": mean_absolute_error(actuals, pred),
    "MSE": mean_squared_error(actuals, pred),
    "RMSE": np.sqrt(mean_squared_error(actuals, pred)),
    "MAPE": mean_absolute_percentage_error(actuals, pred) * 100,  # Convert to percentage
    "MASE": mase(actuals, pred, train[target].values),
    "Forecast Bias": forecast_bias(actuals, pred),
}

# Print metrics in a formatted way
print(f"Performance metrics for {lstm_config.rnn_type} model:")
print(f"MAE: {metrics['MAE']:.2f} ventilator beds")
print(f"RMSE: {metrics['RMSE']:.2f} ventilator beds")
print(f"MAPE: {metrics['MAPE']:.2f}%")
print(f"MASE: {metrics['MASE']:.4f}")
print(f"Forecast Bias: {metrics['Forecast Bias']:.2f} ventilator beds")

# %%
# --- Create forecast visualization with Matplotlib ---
# Create a DataFrame with actual and predicted values
pred_df = pd.DataFrame({"LSTM Forecast": pred}, index=test.index)
pred_df = test[[target]].join(pred_df)

# Create the forecast plot
plt.figure(figsize=(12, 6))
plt.plot(pred_df.index, pred_df[target], 'b-', linewidth=2, label='Actual')
plt.plot(pred_df.index, pred_df['LSTM Forecast'], 'r-', linewidth=2, label='LSTM with Enhanced Features')
plt.title(f"COVID-19 Ventilator Bed Occupancy: Actual vs Forecast\nMAE: {metrics['MAE']:.2f} | RMSE: {metrics['RMSE']:.2f} | MAPE: {metrics['MAPE']:.2f}% | MASE: {metrics['MASE']:.4f}")
plt.xlabel('Date')
plt.ylabel('Ventilator Beds')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../figures/lstm_forecast.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# --- Plot residuals (error analysis) ---
residuals = actuals - pred
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(residuals, kde=True)
plt.title("Residuals Distribution")
plt.xlabel("Residual")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.scatter(pred, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuals vs Predicted Values")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.tight_layout()
plt.savefig('../figures/residuals.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot residuals over time
plt.figure(figsize=(15, 5))
plt.plot(test.index, residuals, label='Residuals')
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuals Over Time")
plt.xlabel("Date")
plt.ylabel("Residual")
plt.legend()
plt.tight_layout()
plt.savefig('../figures/residuals_time.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# --- Feature Importance Analysis ---
# Analyze feature importance by perturbing the input data
feature_importance = {}

# Select a subset of features to analyze to save time
features_to_analyze = final_features[:15] if len(final_features) >= 15 else final_features  # Analyze top 15 features

try:
    for feature in features_to_analyze:
        # Create a copy of the test data
        test_perturbed = test.copy()
        
        # Shuffle the feature values
        test_perturbed[feature] = np.random.permutation(test_perturbed[feature].values)
        
        # Handle potential inf or NaN values
        test_perturbed.replace([np.inf, -np.inf], np.nan, inplace=True)
        test_perturbed.fillna(test_perturbed.mean(), inplace=True)
        
        # Prepare data for prediction
        perturbed_data = selected_df.copy()
        perturbed_data.loc[test_perturbed.index, feature] = test_perturbed[feature]
        
        # Ensure no inf or NaN values
        perturbed_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        perturbed_data.fillna(perturbed_data.mean(), inplace=True)
        
        # Create a new datamodule with the perturbed data
        try:
            perturbed_datamodule = TimeSeriesDataModule(
                data=perturbed_data,
                n_val=val.shape[0],
                n_test=test.shape[0],
                window=window_size,
                horizon=horizon,
                normalize="global",
                batch_size=32,
                num_workers=2,  # Reduced to avoid potential memory issues
            )
            perturbed_datamodule.setup()
            
            # Make predictions with the perturbed data
            perturbed_pred = trainer.predict(lstm_model, perturbed_datamodule.test_dataloader())
            perturbed_pred = torch.cat(perturbed_pred).squeeze().detach().numpy()
            perturbed_pred = perturbed_pred * perturbed_datamodule.train.std + perturbed_datamodule.train.mean
            
            # Calculate the increase in error due to perturbation
            original_mse = mean_squared_error(actuals, pred)
            perturbed_mse = mean_squared_error(actuals, perturbed_pred)
            
            # Feature importance is the percentage increase in error
            importance = (perturbed_mse - original_mse) / original_mse * 100
            feature_importance[feature] = importance
            print(f"Analyzed importance of {feature}: {importance:.2f}%")
        except Exception as e:
            print(f"Error analyzing feature {feature}: {e}")
            feature_importance[feature] = 0  # Assign zero importance to problematic features
except Exception as e:
    print(f"Feature importance analysis failed with error: {e}")
    print("Skipping detailed feature importance analysis")

# Only plot feature importance if we have data
if feature_importance:
    # Sort features by importance
    feature_importance = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)}
    
    # Plot feature importance
    plt.figure(figsize=(14, 8))
    plt.barh(list(feature_importance.keys()), list(feature_importance.values()))
    plt.xlabel('Percentage Increase in MSE')
    plt.title('Feature Importance: Impact on Model Performance')
    plt.tight_layout()
    plt.savefig('../figures/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("No feature importance data to visualize.")

# %%
# --- Model Comparison: LSTM vs. Baseline Models ---
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import time

print("Starting baseline model preparation...")

# Prepare lagged features for baseline models (with smaller lag window for faster execution)
def create_features(df, target, lag=7):
    print(f"Creating {lag} lag features...")
    X = df.copy()
    # Add lags
    for i in range(1, lag + 1):
        X[f'{target}_lag_{i}'] = X[target].shift(i)
    
    # Drop rows with NaN values
    X = X.dropna()
    print(f"Created features dataframe with shape: {X.shape}")
    return X

# Use a smaller lag window for faster execution if needed
lag_window = min(window_size, 7)  # Limit to 7 lags maximum for faster execution
print(f"Using lag window of {lag_window} days")

# Create training and test sets for baseline models
t0 = time.time()
print("Creating training features...")
train_baseline = create_features(train.reset_index(), target, lag=lag_window)
print("Creating test features...")
test_baseline = create_features(test.reset_index(), target, lag=lag_window)
print(f"Feature creation completed in {time.time() - t0:.2f} seconds")

# Features (X) and target (y)
X_train_baseline = train_baseline.drop(['date', target], axis=1)
y_train_baseline = train_baseline[target]
X_test_baseline = test_baseline.drop(['date', target], axis=1)
y_test_baseline = test_baseline[target]
print(f"Training data shape: {X_train_baseline.shape}, Test data shape: {X_test_baseline.shape}")

# Standardize features
print("Standardizing features...")
scaler = StandardScaler()
X_train_baseline_scaled = scaler.fit_transform(X_train_baseline)
X_test_baseline_scaled = scaler.transform(X_test_baseline)

# Train baseline models with timing
baseline_models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    # Reduce estimators for faster execution
    'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=SEED, n_jobs=-1)
}

baseline_predictions = {}
baseline_metrics = {}

for name, model in baseline_models.items():
    print(f"Training {name}...")
    t0 = time.time()
    
    # Train model
    model.fit(X_train_baseline_scaled, y_train_baseline)
    
    # Make predictions
    y_pred = model.predict(X_test_baseline_scaled)
    baseline_predictions[name] = y_pred
    
    # Calculate metrics
    baseline_metrics[name] = {
        'MAE': mean_absolute_error(y_test_baseline, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test_baseline, y_pred)),
        'MAPE': mean_absolute_percentage_error(y_test_baseline, y_pred) * 100
    }
    
    # Add MASE and Forecast Bias if they're not causing problems
    try:
        baseline_metrics[name]['MASE'] = mase(y_test_baseline, y_pred, train[target].values)
        baseline_metrics[name]['Forecast Bias'] = forecast_bias(y_test_baseline, y_pred)
    except Exception as e:
        print(f"Warning: Could not calculate MASE or Forecast Bias: {e}")
    
    print(f"  {name} trained in {time.time() - t0:.2f} seconds")
    print(f"  {name} MAE: {baseline_metrics[name]['MAE']:.2f}")

print("Baseline model training complete!")

# %%
# --- Visualize Model Comparison ---
if baseline_predictions:
    # Add baseline predictions to the visualization
    for name, y_pred in baseline_predictions.items():
        pred_df[name] = np.nan
        
        # Handle index differences safely
        common_indices = pred_df.index.intersection(test_baseline['date'])
        if len(common_indices) > 0:
            # Create an index-matching dictionary
            index_map = {date: i for i, date in enumerate(test_baseline['date'])}
            
            # Update values only for matching dates
            for date in common_indices:
                if date in index_map:
                    idx = index_map[date]
                    if idx < len(y_pred):  # Ensure we don't go out of bounds
                        pred_df.loc[date, name] = y_pred[idx]
        else:
            print(f"Warning: No matching indices found for {name} predictions")
    
    # Create a comparison plot (Matplotlib version)
    plt.figure(figsize=(15, 8))
    
    # Plot actual values
    plt.plot(pred_df.index, pred_df[target], 'b-', linewidth=2.5, label='Actual')
    
    # Plot LSTM forecast
    plt.plot(pred_df.index, pred_df['LSTM Forecast'], 'r-', linewidth=2, label='LSTM')
    
    # Plot baseline model forecasts
    colors = ['g', 'c', 'm', 'y', 'k']
    for i, (name, _) in enumerate(baseline_predictions.items()):
        plt.plot(pred_df.index, pred_df[name], color=colors[i % len(colors)], 
                 linestyle='--', linewidth=1.5, label=name)
    
    plt.title('COVID-19 Ventilator Bed Occupancy: Model Comparison')
    plt.xlabel('Date')
    plt.ylabel('Ventilator Beds')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../figures/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create metrics comparison table
    model_metrics = {
        'LSTM': metrics,
        **{name: mets for name, mets in baseline_metrics.items()}
    }
    
    # Extract specific metrics for comparison
    metrics_df = pd.DataFrame({
        'Model': list(model_metrics.keys()),
        'MAE': [m.get('MAE', None) for m in model_metrics.values()],
        'RMSE': [m.get('RMSE', None) for m in model_metrics.values()],
        'MAPE': [m.get('MAPE', None) for m in model_metrics.values()],
        'MASE': [m.get('MASE', None) for m in model_metrics.values()]
    })
    
    print("Model Comparison Metrics:")
    print(metrics_df)

# %%
# --- Setting Up Multi-Step Forecasting ---
# Create a TimeSeriesDataModule for the model with longer horizon
horizon = 7  # Predict a week ahead

# Recreate your datamodule with the new horizon
datamodule = TimeSeriesDataModule(
    data=selected_df,
    n_val=val.shape[0],
    n_test=test.shape[0],
    window=window_size, 
    horizon=horizon,
    normalize="global",
    batch_size=32,
)
datamodule.setup()

# First, check what horizon your datamodule is using
horizon = datamodule.horizon
print(f"Current datamodule horizon: {horizon}")

# Create the properly configured encoder config
encoder_config = RNNConfig(
    input_size=actual_input_size,
    hidden_size=64,
    num_layers=2,
    bidirectional=True
)

# Create the decoder config that works with the FC decoder
decoder_config = {
    "window_size": window_size,
    "horizon": horizon,
    "input_size": actual_input_size
}

# Create the Seq2Seq config using "FC" as decoder_type
seq2seq_config = {
    "encoder_type": "LSTM",
    "decoder_type": "FC",
    "encoder_params": encoder_config.__dict__,
    "decoder_params": decoder_config,
    "decoder_use_all_hidden": True,
    "teacher_forcing_ratio": 0.0,
    "learning_rate": 0.001,
    "input_size": actual_input_size,
    "optimizer_params": {},
    "lr_scheduler_params": {},
    "lr_scheduler": None
}

# Convert to OmegaConf for compatibility
from omegaconf import OmegaConf
seq2seq_config = OmegaConf.create(seq2seq_config)

# We need to modify the forward method of the model to fix the dimension mismatch
class CustomSeq2SeqModel(Seq2SeqModel):
    def forward(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        o, h = self.encoder(x)
        
        if self.hparams.decoder_type == "FC":
            if self.hparams.decoder_use_all_hidden:
                # Reshape output for multi-step prediction
                y_hat = self.decoder(o.reshape(o.size(0), -1))
                # Reshape to match target dimensions: [batch, horizon, 1]
                y_hat = y_hat.view(y_hat.size(0), -1, 1)
            else:
                y_hat = self.decoder(o[:, -1, :])
                y_hat = y_hat.view(y_hat.size(0), -1, 1)
        else:
            # Original code for RNN decoder
            y_hat = torch.zeros_like(y, device=y.device)
            dec_input = x[:, -1:, :]
            for i in range(y.size(1)):
                out, h = self.decoder(dec_input, h)
                out = self.fc(out)
                y_hat[:, i, :] = out.squeeze(1)
                # decide if we are going to use teacher forcing or not
                teacher_force = random.random() < self.hparams.teacher_forcing_ratio
                if teacher_force:
                    dec_input = y[:, i, :].unsqueeze(1)
                else:
                    dec_input = out
                    
        return y_hat, y

# Create the custom model
seq2seq_model = CustomSeq2SeqModel(seq2seq_config)

# %%
# --- Training the Seq2Seq Model for Multi-Step Forecasting ---
# Train with reduced epochs for faster experimentation
trainer = pl.Trainer(
    min_epochs=5,
    max_epochs=100,
    devices=1, 
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    log_every_n_steps=10,
    callbacks=[
        pl.callbacks.EarlyStopping(monitor="valid_loss", patience=10, mode="min"),
    ]
)

# Try training the model
try:
    trainer.fit(seq2seq_model, datamodule)
    print("Training completed successfully!")
except Exception as e:
    print(f"Error during training: {e}")

# %%
# --- Evaluate Multi-Step Predictions ---
# For multi-step predictions
pred = trainer.predict(seq2seq_model, datamodule.test_dataloader())
pred = torch.cat(pred).detach().numpy()

print(f"Raw prediction shape: {pred.shape}")

# Denormalize predictions
pred = pred * datamodule.train.std + datamodule.train.mean

# Get actual values from test set
actuals = test[target].values
print(f"Actuals shape: {actuals.shape}, length: {len(actuals)}")
print(f"Predictions shape: {pred.shape}, length: {len(pred)}")

# Reshape predictions to match the expected format
# We're changing from (130, 7, 1) to (130*7)
flattened_pred = pred.reshape(-1)

# Now we need to create the corresponding actual values
# For each of the 130 samples, we need the next 7 values from actuals
horizon_predictions = []
horizon_actuals = []

# For each sample in the test set up to the limit of our predictions
for i in range(min(len(pred), len(actuals) - horizon + 1)):
    # Add the prediction for this sample (all horizons)
    horizon_predictions.append(pred[i].flatten())
    
    # Add the actual values for the next 'horizon' steps
    horizon_actuals.append(actuals[i:i+horizon])

# Convert to numpy arrays
horizon_predictions = np.array(horizon_predictions)
horizon_actuals = np.array(horizon_actuals)

print(f"Reshaped predictions: {horizon_predictions.shape}")
print(f"Reshaped actuals: {horizon_actuals.shape}")

# Calculate overall metrics
if len(horizon_predictions) > 0:
    # Overall metrics across all horizons
    mae_overall = mean_absolute_error(horizon_actuals.flatten(), horizon_predictions.flatten())
    rmse_overall = np.sqrt(mean_squared_error(horizon_actuals.flatten(), horizon_predictions.flatten()))
    print(f"Overall MAE: {mae_overall:.2f}")
    print(f"Overall RMSE: {rmse_overall:.2f}")
    
    # Metrics for each forecast horizon
    for h in range(horizon):
        pred_h = horizon_predictions[:, h]
        actual_h = horizon_actuals[:, h]
        
        mae_h = mean_absolute_error(actual_h, pred_h)
        rmse_h = np.sqrt(mean_squared_error(actual_h, pred_h))
        
        # Handle potential division by zero in MAPE calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            mape_h = np.mean(np.abs((actual_h - pred_h) / actual_h)) * 100
            mape_h = np.nan_to_num(mape_h, nan=0.0)  # Replace NaN with 0
        
        print(f"Horizon {h+1} - MAE: {mae_h:.2f}, RMSE: {rmse_h:.2f}, MAPE: {mape_h:.2f}%")
else:
    print("Not enough prediction samples to calculate metrics.")

# %%
# --- Visualize Multi-Step Forecasting Results ---
# Create a dataframe to hold our forecasts and actual values
forecast_df = pd.DataFrame(index=test.index[:len(horizon_predictions)])

# Add actual values
forecast_df[target] = test[target].values[:len(horizon_predictions)]

# Add predictions for each horizon
for h in range(horizon):
    forecast_df[f'Horizon_{h+1}'] = np.nan
    
    # Fill in predictions at the appropriate time points
    for i in range(len(horizon_predictions)):
        if i + h < len(forecast_df):
            forecast_df.iloc[i + h, forecast_df.columns.get_loc(f'Horizon_{h+1}')] = horizon_predictions[i, h]

# Create Matplotlib plot for the forecast
plt.figure(figsize=(15, 8))

# Plot actual values
plt.plot(forecast_df.index, forecast_df[target], color='blue', linewidth=2.5, label='Actual')

# Plot predictions for each horizon with different colors
colors = plt.cm.tab10(np.linspace(0, 1, horizon))
for h in range(horizon):
    plt.plot(forecast_df.index, forecast_df[f'Horizon_{h+1}'], 
             color=colors[h], linewidth=1.5, linestyle='--', 
             label=f'Horizon {h+1}')

plt.title('Multi-horizon COVID-19 Ventilator Bed Occupancy Forecast')
plt.xlabel('Date')
plt.ylabel('Occupied Ventilator Beds')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../figures/multi_horizon_forecast.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# --- Create a visualization for multi-step forecasting with selected starting points ---
# Select a few test points for demonstration (every 10 samples)
sample_indices = list(range(0, len(horizon_predictions), 10))

# Set up the figure
plt.figure(figsize=(15, 10))

# Plot the actual values for the entire period
plt.plot(test.index, test[target], color='blue', linewidth=2.5, label='Actual')

# Add multi-step forecasts for selected starting points
colors = plt.cm.tab10(np.linspace(0, 1, len(sample_indices)))

for idx, i in enumerate(sample_indices):
    if i < len(horizon_predictions):
        # Get the date range for this forecast
        start_date = test.index[i]
        if i + horizon <= len(test.index):
            date_range = test.index[i:i+horizon]
        else:
            # In case we're near the end of the test set
            date_range = test.index[i:] 
            
        # Plot this forecast
        plt.plot(date_range, horizon_predictions[i, :len(date_range)], 
                 color=colors[idx], linewidth=1.5, marker='o', markersize=4,
                 label=f'Forecast from {start_date.strftime("%m-%d")}')

plt.title('Multi-step COVID-19 Ventilator Bed Occupancy Forecast')
plt.xlabel('Date')
plt.ylabel('Occupied Ventilator Beds')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../figures/multi_step_sample_forecasts.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# --- CNN-LSTM Model Implementation ---
import torch
import torch.nn as nn
import pytorch_lightning as pl

# Define the CNN-LSTM model class
class CNNLSTMForecastModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size=64, num_layers=2, cnn_filters=32, 
                 kernel_size=3, dropout=0.2, learning_rate=0.001, horizon=7):
        super().__init__()
        self.save_hyperparameters()
        
        # CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=cnn_filters, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1),
            nn.Conv1d(in_channels=cnn_filters, out_channels=cnn_filters*2, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(dropout)
        )
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=cnn_filters*2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layer for multi-step output
        self.fc = nn.Linear(hidden_size*2, horizon)  # *2 for bidirectional
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN expects [batch, channels, length]
        # x is [batch, sequence, features] -> transpose to [batch, features, sequence]
        x_cnn = x.transpose(1, 2)
        
        # Apply CNN
        cnn_out = self.cnn(x_cnn)
        
        # Transpose back to [batch, sequence, features] for LSTM
        lstm_in = cnn_out.transpose(1, 2)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(lstm_in)
        
        # Use the last time step for prediction
        lstm_out = lstm_out[:, -1, :]
        
        # Generate multi-step forecast
        output = self.fc(lstm_out)
        
        # Reshape to [batch, horizon, 1] for consistency with other models
        return output.view(batch_size, -1, 1)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('valid_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('test_loss', loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

# %%
# --- Create the CNN-LSTM model ---
cnn_lstm_model = CNNLSTMForecastModel(
    input_size=actual_input_size,
    hidden_size=64,
    num_layers=5,
    cnn_filters=32,
    kernel_size=3,
    dropout=0.2,
    learning_rate=0.0001,
    horizon=horizon
)

# Train with early stopping
trainer = pl.Trainer(
    min_epochs=5,
    max_epochs=100,
    devices=1, 
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    log_every_n_steps=10,
    callbacks=[
        pl.callbacks.EarlyStopping(monitor="valid_loss", patience=10)
    ]
)

# %%
# --- Train the CNN-LSTM model ---
try:
    trainer.fit(cnn_lstm_model, datamodule)
    print("Training completed successfully!")
except Exception as e:
    print(f"Error during training: {e}")
    import traceback
    traceback.print_exc()

# %%
# --- Evaluate CNN-LSTM Model ---
cnn_lstm_model.eval()

# Function to make predictions with proper error handling
def make_predictions(model, dataloader):
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            # Forward pass
            pred = model(x)
            # Save predictions and targets
            all_predictions.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    return predictions, targets

# Make predictions
try:
    predictions, targets = make_predictions(cnn_lstm_model, datamodule.test_dataloader())
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")
    
    # Denormalize predictions and targets
    predictions = predictions * datamodule.train.std + datamodule.train.mean
    targets = targets * datamodule.train.std + datamodule.train.mean
    
    # Calculate metrics
    # Overall metrics
    mae = mean_absolute_error(targets.flatten(), predictions.flatten())
    rmse = np.sqrt(mean_squared_error(targets.flatten(), predictions.flatten()))
    
    # Avoid division by zero in MAPE calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((targets.flatten() - predictions.flatten()) / targets.flatten())) * 100
        mape = np.nan_to_num(mape)
    
    print(f"\nOverall Metrics:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Metrics by horizon
    print("\nMetrics by Forecast Horizon:")
    for h in range(predictions.shape[1]):
        h_mae = mean_absolute_error(targets[:, h, 0], predictions[:, h, 0])
        h_rmse = np.sqrt(mean_squared_error(targets[:, h, 0], predictions[:, h, 0]))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            h_mape = np.mean(np.abs((targets[:, h, 0] - predictions[:, h, 0]) / targets[:, h, 0])) * 100
            h_mape = np.nan_to_num(h_mape)
        
        print(f"Horizon {h+1} - MAE: {h_mae:.2f}, RMSE: {h_rmse:.2f}, MAPE: {h_mape:.2f}%")
except Exception as e:
    print(f"Error during prediction: {e}")
    import traceback
    traceback.print_exc()

# %%
# --- Clean up any artifacts created during training ---
if os.path.exists("lightning_logs"):
    shutil.rmtree("lightning_logs")

# %%
# --- Create CNN-LSTM forecast visualization ---
# Create a dataframe to hold our forecasts and actual values
forecast_df = pd.DataFrame(index=test.index[:len(predictions)])

# Add actual values
forecast_df[target] = test[target].values[:len(forecast_df)]

# Add predictions for each horizon
for h in range(horizon):
    forecast_df[f'Horizon_{h+1}'] = np.nan
    
    # Fill in predictions at the appropriate time points
    for i in range(len(predictions)):
        if i + h < len(forecast_df):
            forecast_df.iloc[i + h, forecast_df.columns.get_loc(f'Horizon_{h+1}')] = predictions[i, h, 0]

# Create a plot using Matplotlib
plt.figure(figsize=(15, 8))

# Add actual values
plt.plot(forecast_df.index, forecast_df[target], color='blue', linewidth=2.5, label='Actual')

# Add prediction for each horizon with different colors
colors = plt.cm.tab10(np.linspace(0, 1, horizon))
for h in range(horizon):
    plt.plot(forecast_df.index, forecast_df[f'Horizon_{h+1}'], 
             color=colors[h], linewidth=1.5, linestyle='--',
             label=f'Horizon {h+1}')

plt.title('CNN-LSTM COVID-19 Ventilator Bed Occupancy Forecast by Horizon')
plt.xlabel('Date')
plt.ylabel('Occupied Ventilator Beds')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../figures/cnn_lstm_forecast_by_horizon.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# --- Comparative Analysis of Models ---
# Create a comparison table of the models
comparison_data = {
    'Model': ['LSTM (Single-Step)', 'Seq2Seq (Multi-Step)', 'CNN-LSTM (Multi-Step)'],
    'MAE': [metrics['MAE'], mae_overall, mae],
    'RMSE': [metrics['RMSE'], rmse_overall, rmse],
    'MAPE': [metrics['MAPE'], None, mape],  # Missing MAPE for Seq2Seq
}

comparison_df = pd.DataFrame(comparison_data)
print("Model Performance Comparison:")
print(comparison_df)

# Create a bar chart for model comparison
plt.figure(figsize=(12, 8))
models = comparison_df['Model']
x = np.arange(len(models))
width = 0.3

plt.bar(x - width/2, comparison_df['MAE'], width, label='MAE')
plt.bar(x + width/2, comparison_df['RMSE'], width, label='RMSE')

plt.xlabel('Model')
plt.ylabel('Error Metric')
plt.title('Performance Comparison of Different Models')
plt.xticks(x, models, rotation=15)
plt.legend(loc='best')
plt.grid(True, alpha=0.3, axis='y')

for i, v in enumerate(comparison_df['MAE']):
    plt.text(i - width/2, v + 0.5, f'{v:.2f}', ha='center')
    
for i, v in enumerate(comparison_df['RMSE']):
    plt.text(i + width/2, v + 0.5, f'{v:.2f}', ha='center')

plt.tight_layout()
plt.savefig('../figures/model_comparison_chart.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# --- Save the models for future use ---
# Save the models to disk
os.makedirs("../models", exist_ok=True)

torch.save(lstm_model.state_dict(), '../models/lstm_model.pth')
torch.save(seq2seq_model.state_dict(), '../models/seq2seq_model.pth')
torch.save(cnn_lstm_model.state_dict(), '../models/cnn_lstm_model.pth')

print("Models saved successfully!")

# %%
# --- End of Experiment ---
print("Experiment completed successfully!")

# Clean up any artifacts created during training
import shutil
if os.path.exists("lightning_logs"):
    shutil.rmtree("lightning_logs")
