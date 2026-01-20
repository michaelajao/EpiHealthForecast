"""
Feature Engineering Module for COVID-19 Hospitalization Forecasting

This module provides comprehensive feature engineering functions for time series
forecasting of COVID-19 mechanical ventilator bed demand.

Features include:
- Lockdown period indicators and interactions
- Rate of change and momentum features
- Ratio features (hospitalization to ventilator ratios)
- Calendar features (weekends, holidays, seasons)
- Peak-related features (days since/until peaks)
- Wave indicators
- Lag and rolling statistics features

Reference: Oyelakin et al. (2023) "Deep Learning-Based COVID-19 Hospitalization Forecasting"
           IEEE ICMLA 2023, Jacksonville, FL, USA
"""

import pandas as pd
import numpy as np
import holidays
from typing import Dict, List, Tuple, Optional


# Define UK lockdown periods for feature engineering
LOCKDOWN_DATES: Dict[str, Dict[str, str]] = {
    'Lockdown 1': {'start': '2020-03-23', 'end': '2020-07-04'},
    'Lockdown 2': {'start': '2020-11-05', 'end': '2020-12-02'},
    'Lockdown 3': {'start': '2021-01-06', 'end': '2021-04-12'}
}


def calculate_vax_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Vax index based on vaccination rates and efficacy across age groups.
    
    This is a synthetic vaccination index designed to capture the effect of vaccination
    campaigns on hospitalization and ICU admission rates. It accounts for:
    - Age-stratified vaccination rates
    - First and second dose efficacy differences
    - Age-specific ICU admission probabilities
    
    Args:
        df (pd.DataFrame): DataFrame containing 'date' column.
    
    Returns:
        pd.DataFrame: DataFrame with an additional 'Vax_index' column.
    """
    # Constants based on epidemiological literature
    total_population = 60_000_000  # England population
    number_of_age_groups = 5
    
    # Vaccine efficacy estimates (by age group)
    vaccine_efficacy_first_dose = [0.89, 0.427, 0.76, 0.854, 0.75]
    vaccine_efficacy_second_dose = [0.92, 0.86, 0.81, 0.85, 0.80]
    
    # Age-specific ICU admission probabilities
    age_group_probabilities_icu = [0.01, 0.02, 0.05, 0.1, 0.15]
    
    # Vaccination rollout parameters
    monthly_vaccination_rate_increase = 0.05
    vaccination_start_date = pd.Timestamp('2021-01-18')
    
    population_per_age_group = total_population / number_of_age_groups
    vax_index_list = []
    monthly_vaccination_rate = 0.0
    
    for index, row in df.iterrows():
        # Increment vaccination rate monthly after start date
        if row['date'].day == 1 and row['date'] >= vaccination_start_date:
            monthly_vaccination_rate += monthly_vaccination_rate_increase
            monthly_vaccination_rate = min(monthly_vaccination_rate, 1.0)
        
        Si_sum = 0.0
        
        for i in range(number_of_age_groups):
            vaccinated_population = monthly_vaccination_rate * population_per_age_group
            
            # Assume equal split between first and second dose recipients
            aij = vaccinated_population / 2  # First dose
            bij = vaccinated_population / 2  # Second dose
            cij = population_per_age_group - aij - bij  # Unvaccinated
            
            # Calculate effective susceptible population
            S_double_prime_i = (vaccine_efficacy_second_dose[i] * bij +
                               vaccine_efficacy_first_dose[i] * aij)
            Si = aij + bij + cij - S_double_prime_i
            
            # Weight by age-specific ICU probability
            pi = age_group_probabilities_icu[i]
            Si_normalized = Si / population_per_age_group
            Si_sum += pi * Si_normalized
        
        vax_index_list.append(Si_sum)
    
    df['Vax_index'] = vax_index_list
    return df


def add_rolling_statistics(
    data: pd.DataFrame, 
    columns: List[str], 
    windows: List[int] = [7, 14]
) -> pd.DataFrame:
    """
    Add rolling mean and standard deviation statistics for specified columns.
    
    Args:
        data (pd.DataFrame): Input DataFrame.
        columns (List[str]): Columns to compute rolling statistics for.
        windows (List[int]): Window sizes for rolling calculations.
    
    Returns:
        pd.DataFrame: DataFrame with additional rolling statistics columns.
    """
    df = data.copy()
    
    for column in columns:
        for window in windows:
            df[f'{column}_rolling_mean_{window}'] = df[column].rolling(window=window).mean()
            df[f'{column}_rolling_std_{window}'] = df[column].rolling(window=window).std()
    
    return df


def add_lockdown_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add features related to UK COVID-19 lockdown periods.
    
    Features include:
    - Binary indicator for being in any lockdown
    - Binary indicators for each specific lockdown period
    - Days since lockdown start
    - Days until lockdown end
    - Days since last lockdown ended
    
    Args:
        data (pd.DataFrame): DataFrame with 'date' column.
    
    Returns:
        pd.DataFrame: DataFrame with lockdown-related features.
    """
    df = data.copy()
    
    # Initialize general lockdown indicators
    df['in_lockdown'] = 0
    df['days_since_lockdown_start'] = np.nan
    df['days_until_lockdown_end'] = np.nan
    
    # Create features for each lockdown period
    for lockdown_name, period in LOCKDOWN_DATES.items():
        start_date = pd.to_datetime(period['start'])
        end_date = pd.to_datetime(period['end'])
        
        # Binary indicator for this specific lockdown
        col_name = f'in_{lockdown_name.lower().replace(" ", "_")}'
        df[col_name] = ((df['date'] >= start_date) & (df['date'] <= end_date)).astype(int)
        
        # Update general lockdown indicator
        df.loc[(df['date'] >= start_date) & (df['date'] <= end_date), 'in_lockdown'] = 1
        
        # Days since lockdown start
        mask_since = df['date'] >= start_date
        df.loc[mask_since, f'days_since_{lockdown_name.lower().replace(" ", "_")}_start'] = (
            (df.loc[mask_since, 'date'] - start_date).dt.days
        )
        
        # Days until lockdown end (for dates within lockdown)
        mask_until = (df['date'] >= start_date) & (df['date'] <= end_date)
        df.loc[mask_until, f'days_until_{lockdown_name.lower().replace(" ", "_")}_end'] = (
            (end_date - df.loc[mask_until, 'date']).dt.days
        )
        
        # Fill NaN values
        df[f'days_since_{lockdown_name.lower().replace(" ", "_")}_start'] = \
            df[f'days_since_{lockdown_name.lower().replace(" ", "_")}_start'].fillna(-1)
        df[f'days_until_{lockdown_name.lower().replace(" ", "_")}_end'] = \
            df[f'days_until_{lockdown_name.lower().replace(" ", "_")}_end'].fillna(-1)
    
    # Calculate general days since/until lockdown
    for i, row in df.iterrows():
        current_date = row['date']
        
        # Find closest past lockdown start
        past_starts = [(pd.to_datetime(period['start']), name) 
                       for name, period in LOCKDOWN_DATES.items() 
                       if pd.to_datetime(period['start']) <= current_date]
        
        if past_starts:
            closest_past_start = max(past_starts, key=lambda x: x[0])
            df.at[i, 'days_since_lockdown_start'] = (current_date - closest_past_start[0]).days
        else:
            df.at[i, 'days_since_lockdown_start'] = -1
            
        # Find closest future lockdown end (if currently in lockdown)
        future_ends = [(pd.to_datetime(period['end']), name) 
                       for name, period in LOCKDOWN_DATES.items() 
                       if pd.to_datetime(period['start']) <= current_date <= pd.to_datetime(period['end'])]
        
        if future_ends:
            closest_future_end = min(future_ends, key=lambda x: x[0])
            df.at[i, 'days_until_lockdown_end'] = (closest_future_end[0] - current_date).days
        else:
            df.at[i, 'days_until_lockdown_end'] = -1
    
    # Days since last lockdown ended
    df['days_since_last_lockdown'] = -1
    for i, row in df.iterrows():
        current_date = row['date']
        
        past_ends = [(pd.to_datetime(period['end']), name) 
                    for name, period in LOCKDOWN_DATES.items() 
                    if pd.to_datetime(period['end']) < current_date]
        
        if past_ends and df.at[i, 'in_lockdown'] == 0:
            most_recent_end = max(past_ends, key=lambda x: x[0])
            df.at[i, 'days_since_last_lockdown'] = (current_date - most_recent_end[0]).days
    
    return df


def add_rate_of_change_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add rate of change (daily difference and percentage change) features.
    
    Args:
        data (pd.DataFrame): Input DataFrame with key metrics.
    
    Returns:
        pd.DataFrame: DataFrame with rate of change features.
    """
    df = data.copy()
    
    # Rate of change for hospital cases
    if 'hospitalCases' in df.columns:
        df['hospitalCases_daily_change'] = df['hospitalCases'].diff()
        df['hospitalCases_pct_change'] = df['hospitalCases'].pct_change() * 100
    
    # Rate of change for new admissions
    if 'newAdmissions' in df.columns:
        df['newAdmissions_daily_change'] = df['newAdmissions'].diff()
        df['newAdmissions_pct_change'] = df['newAdmissions'].pct_change() * 100
    
    # Rate of change for ventilator usage
    if 'covidOccupiedMVBeds' in df.columns:
        df['vent_daily_change'] = df['covidOccupiedMVBeds'].diff()
        df['vent_pct_change'] = df['covidOccupiedMVBeds'].pct_change() * 100
    
    # Rate of change for confirmed cases
    if 'new_confirmed' in df.columns:
        df['confirmed_daily_change'] = df['new_confirmed'].diff()
        df['confirmed_pct_change'] = df['new_confirmed'].pct_change() * 100
    
    return df


def add_momentum_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add momentum features (multi-day differences) for key metrics.
    
    Args:
        data (pd.DataFrame): Input DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with momentum features.
    """
    df = data.copy()
    
    momentum_columns = ['hospitalCases', 'newAdmissions', 'covidOccupiedMVBeds', 'new_confirmed']
    
    for col in momentum_columns:
        if col in df.columns:
            df[f'{col}_3day_momentum'] = df[col].diff(3)
            df[f'{col}_7day_momentum'] = df[col].diff(7)
    
    return df


def add_ratio_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add ratio features capturing relationships between different metrics.
    
    Args:
        data (pd.DataFrame): Input DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with ratio features.
    """
    df = data.copy()
    
    # Percentage of hospital cases requiring ventilation
    if 'covidOccupiedMVBeds' in df.columns and 'hospitalCases' in df.columns:
        df['pct_cases_ventilated'] = (df['covidOccupiedMVBeds'] / df['hospitalCases']) * 100
        df['vent_to_hospital_ratio'] = df['covidOccupiedMVBeds'] / df['hospitalCases']
    
    # Admission to hospital ratio
    if 'newAdmissions' in df.columns and 'hospitalCases' in df.columns:
        df['admission_to_hospital_ratio'] = df['newAdmissions'] / df['hospitalCases']
    
    # Admission to ventilator ratio (proxy for severity)
    if 'newAdmissions' in df.columns and 'covidOccupiedMVBeds' in df.columns:
        df['admission_to_vent_ratio'] = df['newAdmissions'] / df['covidOccupiedMVBeds']
    
    return df


def add_calendar_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar-based features including weekends, holidays, and seasons.
    
    Args:
        data (pd.DataFrame): DataFrame with 'date' column.
    
    Returns:
        pd.DataFrame: DataFrame with calendar features.
    """
    df = data.copy()
    
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Basic calendar features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['date'].dt.quarter
    
    # Weekend flag
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x in [5, 6] else 0)
    
    # UK holiday flags
    uk_holidays_dict = holidays.UK()
    df['is_holiday'] = df['date'].apply(lambda x: 1 if x in uk_holidays_dict else 0)
    
    # Season indicators (1=Winter, 2=Spring, 3=Summer, 4=Fall)
    df['season'] = df['month'].apply(
        lambda x: 1 if x in [12, 1, 2] else 
                  2 if x in [3, 4, 5] else 
                  3 if x in [6, 7, 8] else 4
    )
    
    # One-hot encode seasons
    season_dummies = pd.get_dummies(df['season'], prefix='season')
    df = pd.concat([df, season_dummies], axis=1)
    
    return df


def add_peak_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add features related to peaks in key metrics.
    
    Args:
        data (pd.DataFrame): Input DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with peak-related features.
    """
    df = data.copy()
    
    def days_since_peak(series: pd.Series) -> np.ndarray:
        """Calculate days since the most recent peak value."""
        result = np.zeros(len(series))
        current_max = series.iloc[0]
        current_max_idx = 0
        
        for i in range(1, len(series)):
            if series.iloc[i] > current_max:
                current_max = series.iloc[i]
                current_max_idx = i
            result[i] = i - current_max_idx
        
        return result
    
    # Days since peak for key metrics
    if 'covidOccupiedMVBeds' in df.columns:
        df['days_since_vent_peak'] = days_since_peak(df['covidOccupiedMVBeds'])
    
    if 'hospitalCases' in df.columns:
        df['days_since_hospital_peak'] = days_since_peak(df['hospitalCases'])
    
    if 'newAdmissions' in df.columns:
        df['days_since_admissions_peak'] = days_since_peak(df['newAdmissions'])
    
    return df


def add_acceleration_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add acceleration features (change in rate of change).
    
    Args:
        data (pd.DataFrame): DataFrame with rate of change features.
    
    Returns:
        pd.DataFrame: DataFrame with acceleration features.
    """
    df = data.copy()
    
    if 'hospitalCases_daily_change' in df.columns:
        df['hospitalCases_acceleration'] = df['hospitalCases_daily_change'].diff()
    
    if 'vent_daily_change' in df.columns:
        df['vent_acceleration'] = df['vent_daily_change'].diff()
    
    if 'newAdmissions_daily_change' in df.columns:
        df['admissions_acceleration'] = df['newAdmissions_daily_change'].diff()
    
    return df


def add_lag_features(
    data: pd.DataFrame, 
    columns: List[str],
    lags: List[int] = [1, 7, 14]
) -> pd.DataFrame:
    """
    Add lag features for specified columns.
    
    Args:
        data (pd.DataFrame): Input DataFrame.
        columns (List[str]): Columns to create lag features for.
        lags (List[int]): Lag periods to create.
    
    Returns:
        pd.DataFrame: DataFrame with lag features.
    """
    df = data.copy()
    
    for col in columns:
        if col in df.columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    return df


def add_wave_indicators(data: pd.DataFrame, threshold_multiplier: float = 1.5) -> pd.DataFrame:
    """
    Add COVID wave indicator features.
    
    Args:
        data (pd.DataFrame): Input DataFrame.
        threshold_multiplier (float): Multiplier for wave threshold.
    
    Returns:
        pd.DataFrame: DataFrame with wave indicator features.
    """
    df = data.copy()
    
    if 'covidOccupiedMVBeds' in df.columns:
        # Use rolling mean to smooth data
        smooth = df['covidOccupiedMVBeds'].rolling(window=14, min_periods=1).mean()
        overall_mean = smooth.mean()
        
        # Mark as wave when above threshold
        df['covid_wave'] = (smooth > (overall_mean * threshold_multiplier)).astype(int)
        
        # Alternative wave indicator using rate of change
        if 'covidOccupiedMVBeds_rolling_mean_7' in df.columns:
            df['wave_momentum'] = (df['covidOccupiedMVBeds_rolling_mean_7'].diff(7) > 0).astype(int)
    
    return df


def add_lockdown_interaction_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction features between lockdown status and other metrics.
    
    Args:
        data (pd.DataFrame): DataFrame with lockdown and metric features.
    
    Returns:
        pd.DataFrame: DataFrame with interaction features.
    """
    df = data.copy()
    
    if 'in_lockdown' in df.columns:
        if 'hospitalCases' in df.columns:
            df['lockdown_hospital_interaction'] = df['in_lockdown'] * df['hospitalCases']
        
        if 'newAdmissions' in df.columns:
            df['lockdown_admission_interaction'] = df['in_lockdown'] * df['newAdmissions']
        
        if 'covidOccupiedMVBeds' in df.columns:
            df['lockdown_vent_interaction'] = df['in_lockdown'] * df['covidOccupiedMVBeds']
        
        # Lockdown effectiveness metrics
        for col in ['hospitalCases', 'newAdmissions', 'covidOccupiedMVBeds']:
            if col in df.columns:
                df[f'{col}_lockdown_effect'] = df[col].diff(7) * df['in_lockdown']
    
    return df


def add_composite_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add composite features that capture complex relationships.
    
    Args:
        data (pd.DataFrame): Input DataFrame with base features.
    
    Returns:
        pd.DataFrame: DataFrame with composite features.
    """
    df = data.copy()
    
    # Combined hospital-admission ratio
    if 'hospitalCases' in df.columns and 'newAdmissions' in df.columns:
        df['composite_hospital_admission_ratio'] = (
            (df['hospitalCases'] + df['newAdmissions']) / df['hospitalCases']
        )
    
    # Delta admission to ventilator
    if 'vent_daily_change' in df.columns and 'newAdmissions_daily_change' in df.columns:
        df['delta_admission_to_ventilator'] = (
            df['vent_daily_change'] - df['newAdmissions_daily_change']
        )
    
    # Ventilator pressure indicator
    if 'vent_pct_change' in df.columns and 'vent_to_hospital_ratio' in df.columns:
        df['ventilator_pressure'] = df['vent_pct_change'] * df['vent_to_hospital_ratio']
    
    # System stress indicator
    if all(col in df.columns for col in 
           ['pct_cases_ventilated', 'hospitalCases_pct_change', 'days_since_admissions_peak']):
        df['system_stress'] = (
            df['pct_cases_ventilated'] * 
            df['hospitalCases_pct_change'].clip(lower=0) * 
            (1 / (df['days_since_admissions_peak'] + 1))
        )
    
    return df


def engineer_all_features(
    df: pd.DataFrame,
    include_vax_index: bool = True,
    rolling_windows: List[int] = [7, 14],
    lag_periods: List[int] = [1, 7, 14]
) -> pd.DataFrame:
    """
    Apply all feature engineering transformations to create a comprehensive feature set.
    
    This is the main function to call for complete feature engineering.
    
    Args:
        df (pd.DataFrame): Input DataFrame with at least 'date' and key COVID metrics.
        include_vax_index (bool): Whether to calculate and include Vax_index.
        rolling_windows (List[int]): Window sizes for rolling statistics.
        lag_periods (List[int]): Lag periods for lag features.
    
    Returns:
        pd.DataFrame: DataFrame with all engineered features.
    """
    data = df.copy()
    
    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(data['date']):
        data['date'] = pd.to_datetime(data['date'])
    
    # Sort by date
    data = data.sort_values('date').reset_index(drop=True)
    
    # Calculate Vax index if requested
    if include_vax_index and 'Vax_index' not in data.columns:
        data = calculate_vax_index(data)
    
    # Define columns for rolling/lag features
    key_columns = ['covidOccupiedMVBeds', 'hospitalCases', 'newAdmissions', 'new_confirmed']
    if 'Vax_index' in data.columns:
        key_columns.append('Vax_index')
    
    # Filter to existing columns
    key_columns = [col for col in key_columns if col in data.columns]
    
    # Apply all feature engineering functions
    data = add_rolling_statistics(data, key_columns, rolling_windows)
    data = add_lockdown_features(data)
    data = add_rate_of_change_features(data)
    data = add_momentum_features(data)
    data = add_ratio_features(data)
    data = add_calendar_features(data)
    data = add_peak_features(data)
    data = add_acceleration_features(data)
    data = add_lag_features(data, key_columns, lag_periods)
    data = add_wave_indicators(data)
    data = add_lockdown_interaction_features(data)
    data = add_composite_features(data)
    
    # Handle NaN and infinite values
    # Forward fill ratio columns
    ratio_cols = [col for col in data.columns if 'ratio' in col.lower()]
    data[ratio_cols] = data[ratio_cols].ffill()
    
    # Replace infinities with NaN then fill
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna(0)
    
    return data


def get_feature_groups() -> Dict[str, List[str]]:
    """
    Return predefined feature groups for model training.
    
    Returns:
        Dict[str, List[str]]: Dictionary mapping group names to feature lists.
    """
    return {
        'base_features': [
            'covidOccupiedMVBeds_rolling_mean_7',
            'hospitalCases',
            'newAdmissions',
            'new_confirmed',
            'Vax_index'
        ],
        'lockdown_features': [
            'in_lockdown',
            'days_since_lockdown_start',
            'days_until_lockdown_end',
            'days_since_last_lockdown',
            'lockdown_hospital_interaction',
            'lockdown_admission_interaction',
            'lockdown_vent_interaction'
        ],
        'advanced_features': [
            'vent_to_hospital_ratio',
            'pct_cases_ventilated',
            'hospitalCases_3day_momentum',
            'covidOccupiedMVBeds_3day_momentum',
            'covidOccupiedMVBeds_lag_1',
            'covidOccupiedMVBeds_lag_7',
            'system_stress',
            'admission_to_vent_ratio'
        ],
        'calendar_features': [
            'year',
            'month',
            'day_of_week',
            'is_weekend',
            'is_holiday'
        ]
    }
