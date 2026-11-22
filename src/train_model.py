"""
AirSense – Model Training Module
Location: src/train_model.py

This module contains modular, beginner-friendly functions for:
1. Loading the cleaned Hyderabad dataset.
2. Engineering previous-day lag features and temporal calendar features.
3. Chronological train-test splitting (80% train, 20% test) to prevent lookahead bias.
4. Building and training a Scikit-Learn Pipeline with SimpleImputer (median)
   and RandomForestRegressor.
5. Evaluating model performance using MAE, RMSE, and R-squared against a Dummy baseline.
6. Serializing the fitted pipeline to models/random_forest_aqi.pkl.
"""

import os
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline

# Criteria pollutant columns to lag
POLLUTANT_COLUMNS = [
    'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
    'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene'
]


def load_cleaned_data(filepath: str) -> pd.DataFrame:
    """
    Load cleaned Hyderabad air quality dataset and parse dates.

    Parameters:
        filepath (str): Path to hyderabad_aqi_cleaned.csv.

    Returns:
        pd.DataFrame: Chronologically sorted DataFrame with datetime Date.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cleaned dataset not found at: {filepath}")
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date').reset_index(drop=True)
    return df


def create_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Engineer lag-1 features and calendar features for next-day AQI prediction.

    RULES ENFORCED:
    - Target is current-day AQI: y(t).
    - Features are strictly from previous day: X(t-1) using shift(1).
    - Current-day pollutant readings are NOT used as features.
    - AQI_Bucket is NOT used as an input feature.
    - Calendar features (Month, Day, Day_of_week, Day_of_year) reflect the prediction date.

    Parameters:
        df (pd.DataFrame): Chronologically sorted DataFrame.

    Returns:
        Tuple[pd.DataFrame, List[str]]: (DataFrame with engineered features, list of feature column names)
    """
    df_feat = df.copy()

    # Create previous-day lag-1 features for AQI and all pollutants
    lag_targets = ['AQI'] + [p for p in POLLUTANT_COLUMNS if p in df_feat.columns]
    feature_cols = []

    for col in lag_targets:
        feat_name = f"{col}_lag1"
        df_feat[feat_name] = df_feat[col].shift(1)
        feature_cols.append(feat_name)

    # Calendar temporal features for the target prediction date
    df_feat['Month'] = df_feat['Date'].dt.month
    df_feat['Day'] = df_feat['Date'].dt.day
    df_feat['Day_of_week'] = df_feat['Date'].dt.dayofweek
    df_feat['Day_of_year'] = df_feat['Date'].dt.dayofyear

    calendar_features = ['Month', 'Day', 'Day_of_week', 'Day_of_year']
    feature_cols.extend(calendar_features)

    # Filter out rows where target AQI is missing (cannot train or evaluate without ground truth)
    df_model = df_feat.dropna(subset=['AQI']).copy().reset_index(drop=True)

    return df_model, feature_cols


def prepare_train_test_split(
    df_model: pd.DataFrame,
    feature_cols: List[str],
    split_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Perform chronological train-test split without random shuffling.

    Parameters:
        df_model (pd.DataFrame): DataFrame with valid target AQI and engineered features.
        feature_cols (List[str]): List of predictor feature names.
        split_ratio (float): Fraction of data used for training (default 0.8).

    Returns:
        Tuple: (X_train, X_test, y_train, y_test, train_df, test_df)
    """
    split_idx = int(len(df_model) * split_ratio)
    train_df = df_model.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df_model.iloc[split_idx:].copy().reset_index(drop=True)

    X_train = train_df[feature_cols]
    y_train = train_df['AQI']
    X_test = test_df[feature_cols]
    y_test = test_df['AQI']

    return X_train, X_test, y_train, y_test, train_df, test_df


def build_rf_pipeline(
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 42
) -> Pipeline:
    """
    Construct a Scikit-Learn Pipeline combining SimpleImputer and RandomForestRegressor.

    Imputer uses median strategy fitted strictly on training data.
    """
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('regressor', RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        ))
    ])
    return pipeline


def train_rf_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Pipeline:
    """Fit the Pipeline on the training data."""
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Evaluate the fitted model on test data using MAE, RMSE, and R2.
    Also trains a DummyRegressor baseline (predicting mean y_train) for benchmark comparison.
    """
    y_pred = pipeline.predict(X_test)
    y_train_pred = pipeline.predict(X_train)

    rf_mae = mean_absolute_error(y_test, y_pred)
    rf_rmse = root_mean_squared_error(y_test, y_pred)
    rf_r2 = r2_score(y_test, y_pred)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = root_mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    # Baseline Dummy Regressor (predicts mean of training set)
    dummy = DummyRegressor(strategy='mean')
    dummy.fit(X_train, y_train)
    y_dummy_pred = dummy.predict(X_test)

    dummy_mae = mean_absolute_error(y_test, y_dummy_pred)
    dummy_rmse = root_mean_squared_error(y_test, y_dummy_pred)
    dummy_r2 = r2_score(y_test, y_dummy_pred)

    metrics = {
        'rf_test_mae': float(rf_mae),
        'rf_test_rmse': float(rf_rmse),
        'rf_test_r2': float(rf_r2),
        'rf_train_mae': float(train_mae),
        'rf_train_rmse': float(train_rmse),
        'rf_train_r2': float(train_r2),
        'dummy_test_mae': float(dummy_mae),
        'dummy_test_rmse': float(dummy_rmse),
        'dummy_test_r2': float(dummy_r2)
    }
    return metrics


def save_model(pipeline: Pipeline, output_path: str) -> None:
    """Save the fitted Scikit-Learn pipeline to a pickle/joblib file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    joblib.dump(pipeline, output_path)


def run_training_pipeline(
    input_path: str,
    model_output_path: str
) -> Tuple[Pipeline, Dict[str, float], List[str]]:
    """
    Execute the complete end-to-end training and evaluation pipeline.
    """
    df = load_cleaned_data(input_path)
    df_model, feature_cols = create_features(df)
    X_train, X_test, y_train, y_test, train_df, test_df = prepare_train_test_split(df_model, feature_cols)

    pipeline = build_rf_pipeline(n_estimators=100, max_depth=10, random_state=42)
    pipeline = train_rf_pipeline(pipeline, X_train, y_train)

    metrics = evaluate_model(pipeline, X_train, y_train, X_test, y_test)
    save_model(pipeline, model_output_path)

    return pipeline, metrics, feature_cols


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cleaned_csv = os.path.join(base_dir, 'dataset', 'hyderabad_aqi_cleaned.csv')
    model_file = os.path.join(base_dir, 'models', 'random_forest_aqi.pkl')

    print(f"Loading data from: {cleaned_csv}")
    pipeline, metrics, features = run_training_pipeline(cleaned_csv, model_file)

    print(f"\nModel pipeline saved successfully to: {model_file}")
    print("\n--- Model Evaluation Summary ---")
    print(f"Random Forest Test MAE:   {metrics['rf_test_mae']:.2f}")
    print(f"Random Forest Test RMSE:  {metrics['rf_test_rmse']:.2f}")
    print(f"Random Forest Test R2:    {metrics['rf_test_r2']:.4f}")
    print(f"Baseline (Mean) Test MAE: {metrics['dummy_test_mae']:.2f}")
    print(f"Baseline (Mean) Test R2:  {metrics['dummy_test_r2']:.4f}")
