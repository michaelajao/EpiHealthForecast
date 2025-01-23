# src/data/data_processing.py

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
import re
import warnings
from pandas.api.types import is_numeric_dtype
from pandas.tseries.frequencies import to_offset
from pandas.tseries import offsets
from time import time

class LogTime:
    """
    Context manager for logging the duration of operations.
    """
    def __enter__(self):
        self.start_time = time()
        print("Starting operation...")
        
    def __exit__(self, type, value, traceback):
        elapsed_time = time() - self.start_time
        print(f"Operation completed in {elapsed_time:.2f} seconds.")

def load_csv_data(data_dir: Path, filename: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    
    Args:
        data_dir (Path): Directory where the CSV file is located.
        filename (str): Name of the CSV file.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    file_path = data_dir / "raw" / filename
    try:
        df = pd.read_csv(file_path, parse_dates=["date"])
        print(f"Loaded data from {file_path}")
        return df
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise

def fill_missing_values(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """
    Fill missing values in DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        method (str): Method to fill missing values ('ffill', 'bfill', etc.).
    
    Returns:
        pd.DataFrame: DataFrame with filled missing values.
    """
    filled_df = df.fillna(method=method)
    print(f"Filled missing values using method: {method}")
    return filled_df

def engineer_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create or transform basic features.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with engineered basic features.
    """
    # Example: Rolling 7-day mean for dailyCases
    if "dailyCases" in df.columns:
        df["dailyCases_rolling7"] = df["dailyCases"].rolling(window=7, min_periods=1).mean()
        print("Added dailyCases_rolling7 feature.")
    
    return df

def preprocess_pipeline(df: pd.DataFrame, fill_method: str = "ffill") -> pd.DataFrame:
    """
    Full preprocessing pipeline: load data, fill missing values, and engineer basic features.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        fill_method (str): Method to fill missing values.
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df = fill_missing_values(df, method=fill_method)
    df = engineer_basic_features(df)
    return df

# Vax Index Calculation Function (As previously defined)
def calculate_vax_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Vax index based on vaccination rates and efficacy across age groups.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'date', 'cumAdmissions', 'cumulativeCases', and other relevant columns.
    
    Returns:
        pd.DataFrame: DataFrame with an additional 'Vax_index' column.
    """
    # Constants
    total_population = 60_000_000
    number_of_age_groups = 5
    vaccine_efficacy_first_dose = [0.89, 0.427, 0.76, 0.854, 0.75]  # Added 5th value
    vaccine_efficacy_second_dose = [0.92, 0.86, 0.81, 0.85, 0.80]  # Replaced None with 0.80
    age_group_probabilities_icu = [0.01, 0.02, 0.05, 0.1, 0.15]
    monthly_vaccination_rate_increase = 0.05
    vaccination_start_date = pd.Timestamp('2021-01-18')
    
    # Population per age group
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
            print(f"Updated monthly vaccination rate to {monthly_vaccination_rate} on {row['date']}")
        
        Si_sum = 0.0
        
        for i in range(number_of_age_groups):
            # Vaccinated population for this age group
            vaccinated_population = monthly_vaccination_rate * population_per_age_group
            
            # Assume half received first dose and half received second dose
            aij = vaccinated_population / 2  # First dose
            bij = vaccinated_population / 2  # Second dose
            cij = population_per_age_group - aij - bij  # Unvaccinated
            
            # Calculate S''i based on vaccine efficacy
            # Ensuring indices match
            S_double_prime_i = (vaccine_efficacy_second_dose[i] * bij +
                                 vaccine_efficacy_first_dose[i] * aij)
            
            # Calculate Si
            Si = aij + bij + cij - S_double_prime_i  # Effective susceptible
            
            # Age-specific probability
            pi = age_group_probabilities_icu[i]
            
            # Normalize Si by total population in age group
            Si_normalized = Si / population_per_age_group
            
            # Weighted sum
            Si_sum += pi * Si_normalized
        
        # Vax index for the day
        vax_index = Si_sum
        vax_index_list.append(vax_index)
    
    # Add Vax index to the dataframe
    df['Vax_index'] = vax_index_list
    print("Calculated Vax_index for all dates.")
    return df

def add_seasonal_rolling_features(df: pd.DataFrame, 
                                  rolls: List[int], 
                                  seasonal_periods: List[int], 
                                  columns: List[str], 
                                  agg_funcs: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Adds seasonal rolling features to the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        rolls (List[int]): Number of roll periods (e.g., 3 for 3-week rolling).
        seasonal_periods (List[int]): Seasonal periods (e.g., 7 for weekly, 30 for monthly).
        columns (List[str]): Columns to apply rolling features on.
        agg_funcs (List[str]): Aggregation functions (e.g., 'mean', 'std').
    
    Returns:
        Tuple[pd.DataFrame, List[str]]: Updated DataFrame and list of added feature names.
    """
    added_features = []
    
    for column in columns:
        for roll in rolls:
            for period in seasonal_periods:
                for func in agg_funcs:
                    roll_column = f"{column}_roll_{roll}_period_{period}_{func}"
                    
                    # Calculate the rolling feature
                    window_size = roll * period
                    rolled = df[column].rolling(window=window_size)
                    if func == 'mean':
                        df[roll_column] = rolled.mean()
                    elif func == 'std':
                        df[roll_column] = rolled.std()
                    else:
                        raise ValueError(f"Unsupported aggregation function: {func}")
                    
                    added_features.append(roll_column)
                    print(f"Added rolling feature: {roll_column}")
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    return df, added_features

def add_lags(df: pd.DataFrame, lags: List[int], column: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Adds lag features to the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        lags (List[int]): List of lag periods.
        column (str): Column to add lags for.
    
    Returns:
        Tuple[pd.DataFrame, List[str]]: Updated DataFrame and list of added feature names.
    """
    added_features = []
    for lag in lags:
        lag_col_name = f"{column}_lag_{lag}"
        df[lag_col_name] = df[column].shift(lag)
        added_features.append(lag_col_name)
        print(f"Added lag feature: {lag_col_name}")
    # Drop rows with NaN values
    df.dropna(inplace=True)
    return df, added_features

# Temporal Features Functions (from temporal_features.py)
def time_features_from_frequency_str(freq_str: str) -> List[str]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.

    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """
    features_by_offsets = {
        offsets.YearBegin: [],
        offsets.YearEnd: [],
        offsets.MonthBegin: [
            "Month",
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
        ],
        offsets.MonthEnd: [
            "Month",
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
        ],
        offsets.Week: [
            "Month",
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
            "Is_month_start",
            "Week",
        ],
        offsets.Day: [
            "Month",
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
            "Is_month_start",
            "Week",
            "Day",
            "Dayofweek",
            "Dayofyear",
        ],
        offsets.BusinessDay: [
            "Month",
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
            "Is_month_start",
            "Week",
            "Day",
            "Dayofweek",
            "Dayofyear",
        ],
        offsets.Hour: [
            "Month",
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
            "Is_month_start",
            "Week",
            "Day",
            "Dayofweek",
            "Dayofyear",
            "Hour",
        ],
        offsets.Minute: [
            "Month",
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
            "Is_month_start",
            "Week",
            "Day",
            "Dayofweek",
            "Dayofyear",
            "Hour",
            "Minute",
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return feature

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}

    The following frequencies are supported:

        Y, YS   - yearly
            alias: A
        M, MS   - monthly
        W       - weekly
        D       - daily
        B       - business days
        H       - hourly
        T       - minutely
            alias: min
    """
    raise RuntimeError(supported_freq_msg)

def make_date(df: pd.DataFrame, date_field: str) -> pd.DataFrame:
    """
    Ensure the `df[date_field]` is of the datetime type.

    Args:
        df (pd.DataFrame): Input DataFrame.
        date_field (str): Column name of the date field.

    Returns:
        pd.DataFrame: DataFrame with `date_field` as datetime.
    """
    field_dtype = df[date_field].dtype
    if not np.issubdtype(field_dtype, np.datetime64):
        df[date_field] = pd.to_datetime(df[date_field], infer_datetime_format=True)
        print(f"Converted {date_field} to datetime.")
    return df

def add_temporal_features(
    df: pd.DataFrame,
    field_name: str,
    frequency: str,
    add_elapsed: bool = True,
    prefix: Optional[str] = None,
    drop: bool = True,
    use_32_bit: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Adds temporal features to the DataFrame based on the date field.

    Args:
        df (pd.DataFrame): DataFrame to add features to.
        field_name (str): Name of the date column.
        frequency (str): Frequency of the date column (e.g., 'D' for daily).
        add_elapsed (bool, optional): Whether to add elapsed time. Defaults to True.
        prefix (Optional[str], optional): Prefix for new columns. Defaults to None.
        drop (bool, optional): Whether to drop the original date column. Defaults to True.
        use_32_bit (bool, optional): Whether to use 32-bit data types. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, List[str]]: Updated DataFrame and list of added feature names.
    """
    df = make_date(df, field_name)
    prefix = (re.sub("[Dd]ate$", "", field_name) if prefix is None else prefix) + "_"
    attr = time_features_from_frequency_str(frequency)
    _32_bit_dtype = "int32"
    added_features = []
    for n in attr:
        if n == "Week":
            continue
        df[prefix + n] = (
            getattr(df[field_name].dt, n.lower()).astype(_32_bit_dtype)
            if use_32_bit
            else getattr(df[field_name].dt, n.lower())
        )
        added_features.append(prefix + n)
        print(f"Added temporal feature: {prefix + n}")
    # Handle 'Week' separately
    if "Week" in attr:
        if hasattr(df[field_name].dt, "isocalendar"):
            week = df[field_name].dt.isocalendar().week
        else:
            week = df[field_name].dt.week
        df.insert(
            3, prefix + "Week", week.astype(_32_bit_dtype) if use_32_bit else week
        )
        added_features.append(prefix + "Week")
        print(f"Added temporal feature: {prefix + 'Week'}")
    if add_elapsed:
        mask = ~df[field_name].isna()
        df[prefix + "Elapsed"] = np.where(
            mask, df[field_name].values.astype(np.int64) // 10**9, None
        )
        if use_32_bit:
            if df[prefix + "Elapsed"].isnull().sum() == 0:
                df[prefix + "Elapsed"] = df[prefix + "Elapsed"].astype("int32")
            else:
                df[prefix + "Elapsed"] = df[prefix + "Elapsed"].astype("float32")
        added_features.append(prefix + "Elapsed")
        print(f"Added temporal feature: {prefix + 'Elapsed'}")
    if drop:
        df.drop(field_name, axis=1, inplace=True)
        print(f"Dropped original date column: {field_name}")
    return df, added_features

def _calculate_fourier_terms(
    seasonal_cycle: np.ndarray, max_cycle: int, n_fourier_terms: int
) -> np.ndarray:
    """
    Calculates Fourier terms for a seasonal cycle.

    Args:
        seasonal_cycle (np.ndarray): Array representing the seasonal cycle.
        max_cycle (int): Maximum cycle value (e.g., 52 for weeks, 12 for months).
        n_fourier_terms (int): Number of Fourier terms to calculate.

    Returns:
        np.ndarray: Array containing sine and cosine Fourier terms.
    """
    sin_X = np.empty((len(seasonal_cycle), n_fourier_terms), dtype="float64")
    cos_X = np.empty((len(seasonal_cycle), n_fourier_terms), dtype="float64")
    for i in range(1, n_fourier_terms + 1):
        sin_X[:, i - 1] = np.sin((2 * np.pi * seasonal_cycle * i) / max_cycle)
        cos_X[:, i - 1] = np.cos((2 * np.pi * seasonal_cycle * i) / max_cycle)
    return np.hstack([sin_X, cos_X])

def add_fourier_features(
    df: pd.DataFrame,
    column_to_encode: str,
    max_value: Optional[int] = None,
    n_fourier_terms: int = 1,
    use_32_bit: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Adds Fourier Terms for the specified seasonal cycle column, like month, week, hour, etc.

    Args:
        df (pd.DataFrame): The dataframe which has the seasonal cycles to be encoded.
        column_to_encode (str): The column name which has the seasonal cycle.
        max_value (Optional[int], optional): The maximum value the seasonal cycle can attain. Defaults to None.
        n_fourier_terms (int, optional): Number of Fourier terms to be added. Defaults to 1.
        use_32_bit (bool, optional): Flag to use float32 or int32 to reduce memory. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, List[str]]: Updated DataFrame and list of added Fourier feature names.
    """
    assert (
        column_to_encode in df.columns
    ), "`column_to_encode` should be a valid column name in the dataframe"
    assert is_numeric_dtype(
        df[column_to_encode]
    ), "`column_to_encode` should have numeric values."
    if max_value is None:
        max_value = df[column_to_encode].max()
        warnings.warn(
            f"Inferring max cycle as {max_value} from the data. This may not be accurate if data is less than a single seasonal cycle."
        )
    fourier_features = _calculate_fourier_terms(
        df[column_to_encode].astype(int).values,
        max_cycle=max_value,
        n_fourier_terms=n_fourier_terms,
    )
    feature_names = [
        f"{column_to_encode}_sin_{i}" for i in range(1, n_fourier_terms + 1)
    ] + [f"{column_to_encode}_cos_{i}" for i in range(1, n_fourier_terms + 1)]
    df[feature_names] = fourier_features
    if use_32_bit:
        df[feature_names] = df[feature_names].astype("float32")
    print(f"Added Fourier features: {', '.join(feature_names)}")
    return df, feature_names

def bulk_add_fourier_features(
    df: pd.DataFrame,
    columns_to_encode: List[str],
    max_values: List[int],
    n_fourier_terms: int = 1,
    use_32_bit: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Adds Fourier Terms for all the specified seasonal cycle columns, like month, week, hour, etc.

    Args:
        df (pd.DataFrame): The dataframe which has the seasonal cycles to be encoded.
        columns_to_encode (List[str]): The column names which have the seasonal cycle.
        max_values (List[int]): The list of maximum values the seasonal cycles can attain in the
            same order as the columns to encode. Defaults to None.
        n_fourier_terms (int, optional): Number of Fourier terms to be added. Defaults to 1.
        use_32_bit (bool, optional): Flag to use float32 or int32 to reduce memory. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, List[str]]: Updated DataFrame and list of added Fourier feature names.
    """
    assert len(columns_to_encode) == len(
        max_values
    ), "`columns_to_encode` and `max_values` should be of the same length."
    added_features = []
    for column_to_encode, max_value in zip(columns_to_encode, max_values):
        df, features = add_fourier_features(
            df,
            column_to_encode,
            max_value,
            n_fourier_terms=n_fourier_terms,
            use_32_bit=use_32_bit,
        )
        added_features += features
    return df, added_features

def add_seasonal_rolling_features(df: pd.DataFrame, 
                                  rolls: List[int], 
                                  seasonal_periods: List[int], 
                                  columns: List[str], 
                                  agg_funcs: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Adds seasonal rolling features to the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        rolls (List[int]): Number of roll periods (e.g., 3 for 3-week rolling).
        seasonal_periods (List[int]): Seasonal periods (e.g., 7 for weekly, 30 for monthly).
        columns (List[str]): Columns to apply rolling features on.
        agg_funcs (List[str]): Aggregation functions (e.g., 'mean', 'std').
    
    Returns:
        Tuple[pd.DataFrame, List[str]]: Updated DataFrame and list of added feature names.
    """
    # Implementation as defined earlier
    # Reuse the function defined above
    return add_seasonal_rolling_features(df, rolls, seasonal_periods, columns, agg_funcs)

def add_lag_features(df: pd.DataFrame, lags: List[int], column: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Adds lag features to the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        lags (List[int]): List of lag periods.
        column (str): Column to add lags for.
    
    Returns:
        Tuple[pd.DataFrame, List[str]]: Updated DataFrame and list of added feature names.
    """
    # Reuse the add_lags function defined above
    return add_lags(df, lags, column)

def add_temporal_and_fourier_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds temporal and Fourier features to the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with temporal and Fourier features added.
    """
    # Add temporal features
    field_name = 'date'
    frequency = 'D'  # Daily frequency
    df, temporal_features = add_temporal_features(
        df,
        field_name,
        frequency,
        add_elapsed=True,
        prefix=None,
        drop=True,
        use_32_bit=False
    )
    print(f"Temporal Features Created: {', '.join(temporal_features)}")
    
    # Add Fourier features
    columns_to_encode = ['Week', 'Month']  # Example columns
    max_values = [52, 12]  # Weeks in a year, Months in a year
    n_fourier_terms = 3
    df, fourier_features = bulk_add_fourier_features(
        df,
        columns_to_encode,
        max_values,
        n_fourier_terms=n_fourier_terms,
        use_32_bit=True
    )
    print(f"Fourier Features Created: {', '.join(fourier_features)}")
    
    return df
