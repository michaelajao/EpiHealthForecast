"""
NHS England COVID-19 Data Collection Script

This script fetches COVID-19 hospitalization data from the NHS England API
and processes it for use in forecasting models.

Data includes:
- COVID-19 occupied mechanical ventilator beds
- Hospital admissions
- Hospital cases
- New and cumulative confirmed cases
- Death statistics

Reference: Oyelakin et al. (2023) "Deep Learning-Based COVID-19 Hospitalization Forecasting"
           IEEE ICMLA 2023, Jacksonville, FL, USA
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# NHS England COVID-19 API client
try:
    from uk_covid19 import Cov19API
    UK_COVID_AVAILABLE = True
except ImportError:
    UK_COVID_AVAILABLE = False
    print("Warning: uk_covid19 package not installed. Install with: pip install uk-covid19")


# Define the data structure for NHS COVID-19 API queries
NHS_STRUCTURE = {
    "date": "date",
    "areaName": "areaName",
    "areaCode": "areaCode",
    "covidOccupiedMVBeds": "covidOccupiedMVBeds",
    "cumAdmissions": "cumAdmissions",
    "hospitalCases": "hospitalCases",
    "newAdmissions": "newAdmissions",
    "new_confirmed": "newCasesByPublishDate",
    "new_deceased": "newDeaths28DaysByPublishDate",
    "cumulative_confirmed": "cumCasesByPublishDate",
    "cumulative_deceased": "cumDeaths28DaysByPublishDate",
}


def fetch_nhs_region_data(region_name: str = "England") -> pd.DataFrame:
    """
    Fetch COVID-19 data for a specific NHS region from the UK COVID-19 API.
    
    Args:
        region_name (str): Name of the NHS region (e.g., "England", "London").
    
    Returns:
        pd.DataFrame: DataFrame containing COVID-19 statistics.
    """
    if not UK_COVID_AVAILABLE:
        raise ImportError("uk_covid19 package is required. Install with: pip install uk-covid19")
    
    # Define filter for the region
    region_filter = ["areaType=nhsRegion", f"areaName={region_name}"]
    
    # Create API object and fetch data
    api = Cov19API(filters=region_filter, structure=NHS_STRUCTURE)
    data = api.get_dataframe()
    
    # Convert date to datetime
    data['date'] = pd.to_datetime(data['date'])
    
    # Sort by date
    data = data.sort_values('date').reset_index(drop=True)
    
    print(f"Fetched {len(data)} records for {region_name}")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")
    
    return data


def fetch_nation_data(nation: str = "England") -> pd.DataFrame:
    """
    Fetch COVID-19 data at the nation level from the UK COVID-19 API.
    
    Args:
        nation (str): Name of the nation (e.g., "England", "Wales", "Scotland").
    
    Returns:
        pd.DataFrame: DataFrame containing COVID-19 statistics.
    """
    if not UK_COVID_AVAILABLE:
        raise ImportError("uk_covid19 package is required. Install with: pip install uk-covid19")
    
    # Define filter for nation-level data
    nation_filter = ["areaType=nation", f"areaName={nation}"]
    
    # Create API object and fetch data
    api = Cov19API(filters=nation_filter, structure=NHS_STRUCTURE)
    data = api.get_dataframe()
    
    # Convert date to datetime
    data['date'] = pd.to_datetime(data['date'])
    
    # Sort by date
    data = data.sort_values('date').reset_index(drop=True)
    
    print(f"Fetched {len(data)} records for {nation}")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")
    
    return data


def fetch_all_nhs_regions() -> pd.DataFrame:
    """
    Fetch COVID-19 data for all NHS regions in England.
    
    Returns:
        pd.DataFrame: Combined DataFrame with data from all NHS regions.
    """
    if not UK_COVID_AVAILABLE:
        raise ImportError("uk_covid19 package is required. Install with: pip install uk-covid19")
    
    # NHS England regions
    nhs_regions = [
        "East of England",
        "London",
        "Midlands",
        "North East and Yorkshire",
        "North West",
        "South East",
        "South West"
    ]
    
    all_data = []
    
    for region in nhs_regions:
        print(f"Fetching data for {region}...")
        try:
            region_data = fetch_nhs_region_data(region)
            all_data.append(region_data)
        except Exception as e:
            print(f"Error fetching {region}: {e}")
            continue
    
    # Combine all regions
    combined_data = pd.concat(all_data, ignore_index=True)
    
    print(f"\nTotal records fetched: {len(combined_data)}")
    
    return combined_data


def aggregate_to_england(data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate regional data to England-wide totals.
    
    Args:
        data (pd.DataFrame): DataFrame with regional COVID-19 data.
    
    Returns:
        pd.DataFrame: Aggregated England-level data.
    """
    # Define aggregation rules
    agg_rules = {
        'covidOccupiedMVBeds': 'sum',
        'cumAdmissions': 'sum',
        'hospitalCases': 'sum',
        'newAdmissions': 'sum',
        'new_confirmed': 'sum',
        'new_deceased': 'sum',
        'cumulative_confirmed': 'sum',
        'cumulative_deceased': 'sum',
    }
    
    # Filter to only include numeric columns that exist
    agg_rules = {k: v for k, v in agg_rules.items() if k in data.columns}
    
    # Aggregate by date
    aggregated = data.groupby('date').agg(agg_rules).reset_index()
    
    # Add area name
    aggregated['areaName'] = 'England'
    
    # Sort by date
    aggregated = aggregated.sort_values('date').reset_index(drop=True)
    
    print(f"Aggregated to {len(aggregated)} daily records for England")
    
    return aggregated


def load_population_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load population data for regions.
    
    Args:
        filepath (str, optional): Path to population CSV file.
    
    Returns:
        pd.DataFrame: Population data by region.
    """
    if filepath is None:
        # Default path
        filepath = Path(__file__).parent.parent / "data" / "raw" / "population.csv"
    
    if os.path.exists(filepath):
        population = pd.read_csv(filepath)
        print(f"Loaded population data: {len(population)} regions")
        return population
    else:
        print(f"Warning: Population file not found at {filepath}")
        # Return a default England population
        return pd.DataFrame({
            'areaName': ['England'],
            'population': [56_286_961]  # 2021 estimate
        })


def merge_with_population(data: pd.DataFrame, population: pd.DataFrame) -> pd.DataFrame:
    """
    Merge COVID-19 data with population data.
    
    Args:
        data (pd.DataFrame): COVID-19 data.
        population (pd.DataFrame): Population data.
    
    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    # Merge on area name
    merged = data.merge(population, on='areaName', how='left')
    
    # Fill missing population values
    merged['population'] = merged['population'].fillna(56_286_961)  # England default
    
    return merged


def add_derived_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived metrics to the COVID-19 data.
    
    Args:
        data (pd.DataFrame): Raw COVID-19 data.
    
    Returns:
        pd.DataFrame: Data with additional derived metrics.
    """
    df = data.copy()
    
    # Per capita metrics (per 100,000 population)
    if 'population' in df.columns:
        pop_scale = df['population'] / 100_000
        
        if 'hospitalCases' in df.columns:
            df['hospitalCases_per_100k'] = df['hospitalCases'] / pop_scale
        
        if 'covidOccupiedMVBeds' in df.columns:
            df['mvBeds_per_100k'] = df['covidOccupiedMVBeds'] / pop_scale
        
        if 'new_confirmed' in df.columns:
            df['cases_per_100k'] = df['new_confirmed'] / pop_scale
    
    # 7-day rolling averages
    if 'new_confirmed' in df.columns:
        df['new_confirmed_7day_avg'] = df['new_confirmed'].rolling(7).mean()
    
    if 'newAdmissions' in df.columns:
        df['newAdmissions_7day_avg'] = df['newAdmissions'].rolling(7).mean()
    
    # Case fatality rate (CFR)
    if 'cumulative_deceased' in df.columns and 'cumulative_confirmed' in df.columns:
        df['cfr'] = (df['cumulative_deceased'] / df['cumulative_confirmed']) * 100
    
    return df


def save_data(data: pd.DataFrame, filename: str, output_dir: Optional[str] = None) -> str:
    """
    Save processed data to CSV file.
    
    Args:
        data (pd.DataFrame): Data to save.
        filename (str): Output filename.
        output_dir (str, optional): Output directory path.
    
    Returns:
        str: Path to saved file.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "processed"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / filename
    data.to_csv(output_path, index=False)
    
    print(f"Data saved to: {output_path}")
    return str(output_path)


def collect_and_process_data(save_output: bool = True) -> pd.DataFrame:
    """
    Main function to collect and process all COVID-19 data.
    
    This function:
    1. Fetches data from NHS England API
    2. Aggregates to England level
    3. Adds population data
    4. Adds derived metrics
    5. Optionally saves to CSV
    
    Args:
        save_output (bool): Whether to save the processed data to CSV.
    
    Returns:
        pd.DataFrame: Processed COVID-19 data.
    """
    print("=" * 60)
    print("NHS England COVID-19 Data Collection")
    print("=" * 60)
    
    # Step 1: Fetch data from all NHS regions
    print("\nStep 1: Fetching data from NHS England API...")
    
    if UK_COVID_AVAILABLE:
        raw_data = fetch_all_nhs_regions()
    else:
        # Load from existing file if API not available
        print("API not available, attempting to load from existing file...")
        raw_path = Path(__file__).parent.parent / "data" / "raw" / "nation_data.csv"
        if os.path.exists(raw_path):
            raw_data = pd.read_csv(raw_path)
            raw_data['date'] = pd.to_datetime(raw_data['date'])
        else:
            raise FileNotFoundError("No data source available. Install uk-covid19 package or provide data files.")
    
    # Step 2: Aggregate to England level
    print("\nStep 2: Aggregating to England level...")
    england_data = aggregate_to_england(raw_data)
    
    # Step 3: Add population data
    print("\nStep 3: Adding population data...")
    population = load_population_data()
    england_data = merge_with_population(england_data, population)
    
    # Step 4: Add derived metrics
    print("\nStep 4: Calculating derived metrics...")
    processed_data = add_derived_metrics(england_data)
    
    # Step 5: Save output
    if save_output:
        print("\nStep 5: Saving processed data...")
        save_data(processed_data, "merged_nhs_covid_data.csv")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Data Collection Complete!")
    print("=" * 60)
    print(f"Total records: {len(processed_data)}")
    print(f"Date range: {processed_data['date'].min()} to {processed_data['date'].max()}")
    print(f"Columns: {list(processed_data.columns)}")
    
    return processed_data


def load_existing_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load existing processed COVID-19 data from CSV.
    
    Args:
        filepath (str, optional): Path to CSV file.
    
    Returns:
        pd.DataFrame: Loaded data.
    """
    if filepath is None:
        filepath = Path(__file__).parent.parent / "data" / "processed" / "merged_nhs_covid_data.csv"
    
    data = pd.read_csv(filepath)
    data['date'] = pd.to_datetime(data['date'])
    
    print(f"Loaded {len(data)} records from {filepath}")
    
    return data


if __name__ == "__main__":
    # Run the data collection pipeline
    data = collect_and_process_data(save_output=True)
    print("\nSample of processed data:")
    print(data.head())
