"""
AirSense – Next-Day AQI Prediction Module
Location: src/prediction.py

This module provides reusable, beginner-friendly functions to forecast next-day AQI
using a trained Random Forest pipeline and historical daily observations.

IMPORTANT METHODOLOGICAL & OPERATIONAL NOTES:
1. Pure Historical Causality: The prediction is calculated strictly using
   observations available up to the previous day (t-1). No current-day or future
   telemetry is ever accessed.
2. Missing Lag Handling: The pipeline incorporates a Scikit-Learn SimpleImputer
   (median strategy fitted during training) to handle any missing pollutant readings.
3. Disclaimer: This module operates on historical monitoring records from the
   CPCB dataset. It represents offline retrospective next-day forecasting for
   academic evaluation, NOT a live real-time internet-connected sensor stream.
"""

import os
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd

# Expected feature columns in the exact order the pipeline was trained on
FEATURE_COLUMNS = [
    'AQI_lag1', 'PM2.5_lag1', 'PM10_lag1', 'NO_lag1', 'NO2_lag1',
    'NOx_lag1', 'NH3_lag1', 'CO_lag1', 'SO2_lag1', 'O3_lag1',
    'Benzene_lag1', 'Toluene_lag1', 'Xylene_lag1',
    'Month', 'Day', 'Day_of_week', 'Day_of_year'
]

POLLUTANT_COLS = [
    'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
    'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene'
]


def load_model(model_path: str) -> Any:
    """
    Load the trained Scikit-Learn pipeline from disk.

    Parameters:
        model_path (str): Filepath to the serialized model (.pkl).

    Returns:
        Pipeline: Fitted Scikit-Learn pipeline containing imputer and regressor.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Train the model first via src/train_model.py.")
    pipeline = joblib.load(model_path)
    return pipeline


def get_aqi_category(aqi_value: float) -> Tuple_Category:
    """
    Classify a continuous AQI value into standard CPCB health categories.

    Parameters:
        aqi_value (float): Predicted or actual AQI.

    Returns:
        tuple: (category_name, health_advisory, color_hex)
    """
    if aqi_value <= 50:
        return ("Good", "Minimal health impact. Air quality is considered satisfactory.", "#00b050")
    elif aqi_value <= 100:
        return ("Satisfactory", "Minor breathing discomfort to sensitive people.", "#92d050")
    elif aqi_value <= 200:
        return ("Moderate", "Breathing discomfort to people with asthma and heart diseases.", "#e6b800")
    elif aqi_value <= 300:
        return ("Poor", "Breathing discomfort to most people on prolonged exposure.", "#ff9900")
    elif aqi_value <= 400:
        return ("Very Poor", "Respiratory illness on prolonged exposure.", "#ff0000")
    else:
        return ("Severe", "Affects healthy people and seriously impacts those with existing diseases.", "#990000")


# Type alias for category return
Tuple_Category = tuple


def predict_next_day_aqi(
    historical_df: pd.DataFrame,
    model_path: Optional[str] = None,
    pipeline: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Predict next-day AQI using the latest available historical record.

    Steps:
    1. Validate required columns exist in historical_df.
    2. Sort chronologically by Date.
    3. Extract the most recent daily record to serve as the previous-day lag (t-1).
    4. Compute target date (t = latest_date + 1 day) and extract calendar features.
    5. Construct the 17-feature input vector and run inference through the pipeline.
    6. Return a comprehensive prediction report with category and disclaimer.

    Parameters:
        historical_df (pd.DataFrame): Historical daily dataset.
        model_path (str, optional): Path to serialized model file.
        pipeline (Pipeline, optional): Pre-loaded Scikit-Learn pipeline.

    Returns:
        dict: Detailed prediction results and audit metadata.
    """
    if pipeline is None:
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, 'models', 'random_forest_aqi.pkl')
        pipeline = load_model(model_path)

    # 1. Validation
    required_cols = ['Date', 'AQI'] + POLLUTANT_COLS
    missing_cols = [c for c in required_cols if c not in historical_df.columns]
    if missing_cols:
        raise ValueError(f"Historical DataFrame is missing required columns: {missing_cols}")

    # 2. Chronological sort
    df_sorted = historical_df.copy()
    df_sorted['Date'] = pd.to_datetime(df_sorted['Date'])
    df_sorted = df_sorted.sort_values(by='Date').reset_index(drop=True)

    if len(df_sorted) == 0:
        raise ValueError("Historical DataFrame contains zero records.")

    # 3. Extract latest observation (day t-1)
    latest_record = df_sorted.iloc[-1]
    latest_date = latest_record['Date']
    target_prediction_date = latest_date + pd.Timedelta(days=1)

    # 4. Construct feature dictionary
    feature_dict = {}
    feature_dict['AQI_lag1'] = latest_record['AQI']

    for pol in POLLUTANT_COLS:
        feature_dict[f"{pol}_lag1"] = latest_record[pol]

    feature_dict['Month'] = target_prediction_date.month
    feature_dict['Day'] = target_prediction_date.day
    feature_dict['Day_of_week'] = target_prediction_date.dayofweek
    feature_dict['Day_of_year'] = target_prediction_date.dayofyear

    # 5. Build 1-row DataFrame matching the exact training columns
    input_df = pd.DataFrame([feature_dict])[FEATURE_COLUMNS]

    # Check lag feature availability
    available_lags = input_df.notnull().sum().sum()
    total_lags = input_df.shape[1]

    # 6. Run prediction
    predicted_aqi = float(pipeline.predict(input_df)[0])
    category, advisory, color = get_aqi_category(predicted_aqi)

    result = {
        'target_prediction_date': target_prediction_date.strftime('%Y-%m-%d'),
        'based_on_observation_date': latest_date.strftime('%Y-%m-%d'),
        'predicted_aqi': round(predicted_aqi, 2),
        'aqi_category': category,
        'health_advisory': advisory,
        'category_color': color,
        'feature_availability': f"{available_lags}/{total_lags} features available (missing imputed via training median)",
        'input_lag_summary': {
            'previous_day_aqi': None if pd.isna(latest_record['AQI']) else round(float(latest_record['AQI']), 2),
            'previous_day_pm25': None if pd.isna(latest_record['PM2.5']) else round(float(latest_record['PM2.5']), 2),
            'previous_day_pm10': None if pd.isna(latest_record['PM10']) else round(float(latest_record['PM10']), 2)
        },
        'disclaimer': (
            "NOTICE: This prediction is generated from historical CPCB monitoring records "
            "using previous-day lag features for educational and analytical purposes. "
            "It does NOT represent a live, connected real-time weather station telemetry forecast."
        )
    }

    return result


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cleaned_csv = os.path.join(base_dir, 'dataset', 'hyderabad_aqi_cleaned.csv')
    model_file = os.path.join(base_dir, 'models', 'random_forest_aqi.pkl')

    print(f"Loading historical data from: {cleaned_csv}")
    df_hist = pd.read_csv(cleaned_csv)

    print(f"Running inference using model: {model_file}...")
    pred_result = predict_next_day_aqi(df_hist, model_path=model_file)

    print("\n=== NEXT-DAY AQI PREDICTION RESULT ===")
    for key, value in pred_result.items():
        if key == 'input_lag_summary':
            print(f"  {key}:")
            for sub_k, sub_v in value.items():
                print(f"    - {sub_k}: {sub_v}")
        else:
            print(f"  {key}: {value}")
