"""
Utility functions for loading and preprocessing temperature data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def load_temperature_data(
    filepath: str,
    date_column: str = 'date',
    temp_column: str = 'temperature',
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load temperature data from CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file
    date_column : str
        Name of the date/time column
    temp_column : str
        Name of the temperature column
    start_year : int, optional
        Starting year for filtering data
    end_year : int, optional
        Ending year for filtering data
        
    Returns
    -------
    tuple
        (time_array, temperature_array, dataframe)
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # Ensure date column is datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Extract year
    df['year'] = df[date_column].dt.year
    
    # Filter by year range if specified
    if start_year is not None:
        df = df[df['year'] >= start_year]
    if end_year is not None:
        df = df[df['year'] <= end_year]
    
    # Sort by date
    df = df.sort_values(date_column)
    
    # Convert to decimal years for analysis
    df['decimal_year'] = (
        df['year'] + 
        (df[date_column].dt.dayofyear - 1) / 365.25
    )
    
    # Extract arrays
    time = df['decimal_year'].values
    temperature = df[temp_column].values
    
    return time, temperature, df


def aggregate_to_annual(
    df: pd.DataFrame,
    date_column: str = 'date',
    temp_column: str = 'temperature',
    method: str = 'mean'
) -> pd.DataFrame:
    """
    Aggregate temperature data to annual values.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with temperature data
    date_column : str
        Name of the date column
    temp_column : str
        Name of the temperature column
    method : str
        Aggregation method ('mean', 'median', 'max', 'min')
        
    Returns
    -------
    pd.DataFrame
        Aggregated annual data
    """
    # Ensure date column is datetime
    df[date_column] = pd.to_datetime(df[date_column])
    df['year'] = df[date_column].dt.year
    
    # Aggregate
    if method == 'mean':
        annual = df.groupby('year')[temp_column].mean()
    elif method == 'median':
        annual = df.groupby('year')[temp_column].median()
    elif method == 'max':
        annual = df.groupby('year')[temp_column].max()
    elif method == 'min':
        annual = df.groupby('year')[temp_column].min()
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    
    # Create dataframe
    annual_df = pd.DataFrame({
        'year': annual.index,
        temp_column: annual.values
    })
    
    return annual_df


def detect_outliers(
    temperature: np.ndarray,
    n_sigma: float = 3.0
) -> np.ndarray:
    """
    Detect outliers using the n-sigma method.
    
    Parameters
    ----------
    temperature : np.ndarray
        Temperature values
    n_sigma : float
        Number of standard deviations for outlier threshold
        
    Returns
    -------
    np.ndarray
        Boolean array indicating outliers
    """
    mean = np.mean(temperature)
    std = np.std(temperature)
    
    outliers = np.abs(temperature - mean) > n_sigma * std
    
    return outliers


def detrend_data(
    time: np.ndarray,
    temperature: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    """
    Remove linear trend from temperature data.
    
    Parameters
    ----------
    time : np.ndarray
        Time values
    temperature : np.ndarray
        Temperature values
        
    Returns
    -------
    tuple
        (detrended_temperature, slope, intercept)
    """
    # Fit linear trend
    coeffs = np.polyfit(time, temperature, deg=1)
    slope, intercept = coeffs
    
    # Remove trend
    trend = slope * time + intercept
    detrended = temperature - trend
    
    return detrended, slope, intercept
