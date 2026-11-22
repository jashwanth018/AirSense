"""
AirSense – Data Preprocessing Module
Location: src/preprocessing.py

This module contains modular, beginner-friendly functions for cleaning and
preprocessing the Hyderabad air quality dataset for exploratory data analysis
and machine learning modeling.

Key Preprocessing Principles Applied:
1. Past-only forward filling (ffill with limit=3): Avoids lookahead bias/data leakage.
2. Zero-fill prohibition: Never fills missing pollutant or AQI values with 0.
3. Target preservation: Keeps missing AQI values as NaN (never imputes target).
4. Outlier retention: Identifies IQR extremes without deleting genuine pollution peaks.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Core pollutant columns monitored in ambient air
POLLUTANT_COLUMNS = [
    'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
    'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene'
]


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw CSV air quality data into a Pandas DataFrame.

    Parameters:
        filepath (str): Path to the CSV dataset.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    df = pd.read_csv(filepath)
    return df


def clean_date_and_sort(df: pd.DataFrame, date_column: str = 'Date') -> pd.DataFrame:
    """
    Convert the Date column to datetime64 format, sort chronologically,
    and reset the DataFrame index.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        date_column (str): Name of the date column.

    Returns:
        pd.DataFrame: Chronologically sorted dataframe with datetime index column.
    """
    df_sorted = df.copy()
    df_sorted[date_column] = pd.to_datetime(df_sorted[date_column])
    df_sorted = df_sorted.sort_values(by=date_column).reset_index(drop=True)
    return df_sorted


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Check and remove exact duplicate records if any exist.

    Parameters:
        df (pd.DataFrame): Input dataframe.

    Returns:
        Tuple[pd.DataFrame, int]: (Deduplicated dataframe, count of duplicates removed)
    """
    dup_count = int(df.duplicated().sum())
    df_dedup = df.drop_duplicates().reset_index(drop=True)
    return df_dedup, dup_count


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the count and percentage of missing values for all columns.

    Parameters:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Summary table showing data types, missing counts, and percentages.
    """
    missing_count = df.isnull().sum()
    missing_pct = (df.isnull().mean() * 100).round(2)
    summary = pd.DataFrame({
        'Data Type': df.dtypes,
        'Missing Count': missing_count,
        'Missing Percentage (%)': missing_pct
    })
    return summary


def handle_pollutant_missing_values(
    df: pd.DataFrame,
    pollutants: Optional[List[str]] = None,
    limit: int = 3
) -> pd.DataFrame:
    """
    Handle missing values in pollutant columns using forward filling (ffill).

    RULES ENFORCED:
    - Only past observations are carried forward (temporal causality).
    - limit=3 restricts imputation to a maximum of 3 consecutive missing days.
    - Backward filling (bfill) is NOT used to prevent data leakage.
    - Zero filling is strictly PROHIBITED (0 implies zero pollution).
    - AQI target variable is NOT modified (retains NaN).

    Parameters:
        df (pd.DataFrame): Chronologically sorted dataframe.
        pollutants (list): List of pollutant columns to process.
        limit (int): Maximum consecutive days to forward fill.

    Returns:
        pd.DataFrame: Cleaned dataframe with forward-filled pollutants.
    """
    if pollutants is None:
        pollutants = [col for col in POLLUTANT_COLUMNS if col in df.columns]

    df_cleaned = df.copy()
    # Forward fill up to 'limit' consecutive days only on pollutant predictor features
    df_cleaned[pollutants] = df_cleaned[pollutants].ffill(limit=limit)

    return df_cleaned


def detect_iqr_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Detect statistical outliers using the Interquartile Range (IQR) method:
        IQR = Q3 - Q1
        Lower Bound = Q1 - 1.5 * IQR
        Upper Bound = Q3 + 1.5 * IQR

    NOTE: Genuine pollution spikes (e.g. Diwali, dust storms) are retained
    and NOT automatically dropped.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        columns (list): Columns to evaluate.

    Returns:
        pd.DataFrame: Summary table of IQR thresholds and outlier counts.
    """
    if columns is None:
        columns = [c for c in df.select_dtypes(include=[np.number]).columns]

    records = []
    for col in columns:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        q1 = float(s.quantile(0.25))
        median = float(s.median())
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = s[(s < lower_bound) | (s > upper_bound)]

        records.append({
            'Feature': col,
            'Q1': round(q1, 2),
            'Median': round(median, 2),
            'Q3': round(q3, 2),
            'IQR': round(iqr, 2),
            'Lower Bound': round(lower_bound, 2),
            'Upper Bound': round(upper_bound, 2),
            'Outlier Count': len(outliers),
            'Outlier %': round((len(outliers) / len(s)) * 100, 2),
            'Max Value': round(float(s.max()), 2)
        })

    return pd.DataFrame(records)


def save_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the cleaned DataFrame to a CSV file.

    Parameters:
        df (pd.DataFrame): Cleaned dataframe.
        output_path (str): Destination file path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)


def preprocess_hyderabad_data(
    input_path: str,
    output_path: str,
    ffill_limit: int = 3
) -> Tuple[pd.DataFrame, Dict]:
    """
    Execute the complete end-to-end preprocessing pipeline for Hyderabad data.

    Steps:
    1. Load raw dataset.
    2. Convert Date to datetime and sort chronologically.
    3. Remove any exact duplicate rows.
    4. Handle pollutant missing values using ffill(limit=3).
    5. Keep AQI target and AQI_Bucket unmodified.
    6. Save cleaned dataset to output_path.

    Returns:
        Tuple[pd.DataFrame, Dict]: Cleaned DataFrame and verification metrics dictionary.
    """
    # 1. Load data
    df_raw = load_data(input_path)
    initial_shape = df_raw.shape

    # 2. Date conversion and chronological sort
    df_sorted = clean_date_and_sort(df_raw, date_column='Date')

    # 3. Deduplication check
    df_dedup, dup_count = remove_duplicates(df_sorted)

    # Missing counts before cleaning
    missing_before = int(df_dedup.isnull().sum().sum())
    aqi_missing_before = int(df_dedup['AQI'].isnull().sum())

    # 4. Handle pollutant missing values
    df_cleaned = handle_pollutant_missing_values(df_dedup, limit=ffill_limit)

    # Missing counts after cleaning
    missing_after = int(df_cleaned.isnull().sum().sum())
    aqi_missing_after = int(df_cleaned['AQI'].isnull().sum())

    # 5. Save cleaned dataset
    save_cleaned_data(df_cleaned, output_path)

    metrics = {
        'initial_shape': initial_shape,
        'cleaned_shape': df_cleaned.shape,
        'duplicates_removed': dup_count,
        'total_missing_before': missing_before,
        'total_missing_after': missing_after,
        'aqi_missing_before': aqi_missing_before,
        'aqi_missing_after': aqi_missing_after,
        'date_min': df_cleaned['Date'].min().strftime('%Y-%m-%d'),
        'date_max': df_cleaned['Date'].max().strftime('%Y-%m-%d'),
        'records_retained': len(df_cleaned)
    }

    return df_cleaned, metrics


if __name__ == '__main__':
    # Standalone execution test
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_csv = os.path.join(base_dir, 'dataset', 'hyderabad_aqi_raw.csv')
    cleaned_csv = os.path.join(base_dir, 'dataset', 'hyderabad_aqi_cleaned.csv')

    print(f"Running preprocessing on: {raw_csv}")
    df_clean, stats = preprocess_hyderabad_data(raw_csv, cleaned_csv)

    print("\n=== PREPROCESSING COMPLETED SUCCESSFULLY ===")
    for k, v in stats.items():
        print(f"{k}: {v}")
