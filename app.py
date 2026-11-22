"""
AirSense – Hyderabad Air Quality Analysis and AQI Prediction
Streamlit Web Application (Phase 6)

Sections:
1. Overview      – Key statistics and AQI trend at a glance.
2. Air Quality Analysis – Filtered trend charts and pollutant breakdowns.
3. AQI Prediction – Next-day AQI forecast using the trained Random Forest model.
4. About AirSense – Project information, methodology, and model performance.

Run:
    cd AirSense
    streamlit run app.py
"""

# ---------------------------------------------------------------------------
# 0. Imports and path setup
# ---------------------------------------------------------------------------
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Add the src/ directory to the Python path so we can import project modules.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(APP_DIR, "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

# Project modules (from src/)
from prediction import predict_next_day_aqi, load_model, get_aqi_category

# ---------------------------------------------------------------------------
# 1. Page configuration (must be the FIRST Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AirSense – Hyderabad AQI",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 2. Constants
# ---------------------------------------------------------------------------
DATASET_PATH = os.path.join(APP_DIR, "dataset", "hyderabad_aqi_cleaned.csv")
MODEL_PATH   = os.path.join(APP_DIR, "models", "random_forest_aqi.pkl")
VIZ_DIR      = os.path.join(APP_DIR, "visualizations")

# CPCB standard AQI category color codes
CATEGORY_COLORS = {
    "Good":       "#00b050",
    "Satisfactory": "#92d050",
    "Moderate":   "#e6b800",
    "Poor":       "#ff9900",
    "Very Poor":  "#ff0000",
    "Severe":     "#990000",
}

POLLUTANT_COLS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene",
]

# ---------------------------------------------------------------------------
# 3. Data loading (cached so it runs only once per session)
# ---------------------------------------------------------------------------
@st.cache_data
def load_dataset():
    """Load the cleaned Hyderabad AQI dataset from disk."""
    df = pd.read_csv(DATASET_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 4. Helper utilities
# ---------------------------------------------------------------------------
def category_color(cat: str) -> str:
    """Return hex color for a given AQI category name."""
    return CATEGORY_COLORS.get(cat, "#888888")


def metric_card(label: str, value: str, unit: str = "") -> None:
    """Render a simple bold KPI card using Streamlit markdown."""
    st.markdown(
        f"""
        <div style="background:#1e2a3a;border-radius:10px;padding:18px 14px;text-align:center;">
            <p style="color:#aab8c8;font-size:13px;margin:0 0 6px 0;">{label}</p>
            <p style="color:#ffffff;font-size:26px;font-weight:700;margin:0;">{value}</p>
            <p style="color:#7b90a0;font-size:12px;margin:4px 0 0 0;">{unit}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_viz(filename: str, caption: str = "") -> None:
    """Display a pre-rendered visualisation PNG from the visualizations/ folder."""
    path = os.path.join(VIZ_DIR, filename)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Image not found: {filename}")


# ---------------------------------------------------------------------------
# 5. PAGE 1 – Overview
# ---------------------------------------------------------------------------
def show_overview(df: pd.DataFrame) -> None:
    st.header("🏙️ Hyderabad Air Quality – Overview")
    st.markdown(
        "A snapshot of **historical air quality monitoring** in Hyderabad, "
        "based on CPCB daily records from **January 2015 to July 2020**."
    )

    # --- KPI row ---
    df_aqi = df.dropna(subset=["AQI"])  # only rows where AQI was measured

    total_days      = len(df)
    valid_aqi_days  = len(df_aqi)
    mean_aqi        = df_aqi["AQI"].mean()
    max_aqi         = df_aqi["AQI"].max()
    min_aqi         = df_aqi["AQI"].min()
    top_category    = df_aqi["AQI_Bucket"].mode()[0] if "AQI_Bucket" in df_aqi.columns else "N/A"

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("Monitoring Days",   f"{total_days:,}",       "total records")
    with c2: metric_card("Valid AQI Days",    f"{valid_aqi_days:,}",   "AQI available")
    with c3: metric_card("Average AQI",       f"{mean_aqi:.1f}",       "AQI points")
    with c4: metric_card("Max AQI Recorded",  f"{max_aqi:.0f}",        "peak value")
    with c5: metric_card("Most Common Category", top_category,         "dominant air quality")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Daily AQI trend ---
    st.subheader("📈 Daily AQI Trend (2015–2020)")
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(df_aqi["Date"], df_aqi["AQI"], color="#4c9be8", linewidth=0.8, alpha=0.85)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("AQI", fontsize=11)
    ax.set_title("Daily AQI – Hyderabad (2015–2020)", fontsize=13)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- AQI Category distribution ---
    st.subheader("🗂️ AQI Category Distribution")
    if "AQI_Bucket" in df.columns:
        cat_counts = (
            df_aqi["AQI_Bucket"]
            .value_counts()
            .reindex(["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"])
            .dropna()
        )
        colors = [category_color(c) for c in cat_counts.index]

        fig2, ax2 = plt.subplots(figsize=(9, 4))
        bars = ax2.bar(cat_counts.index, cat_counts.values, color=colors, edgecolor="white", linewidth=0.6)
        ax2.set_xlabel("AQI Category", fontsize=11)
        ax2.set_ylabel("Number of Days", fontsize=11)
        ax2.set_title("Number of Days in Each AQI Category", fontsize=13)
        for bar, val in zip(bars, cat_counts.values):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 4,
                str(int(val)),
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # --- Key insights ---
    st.subheader("💡 Key Insights")
    max_date = df_aqi.loc[df_aqi["AQI"].idxmax(), "Date"].strftime("%d %b %Y")
    min_date = df_aqi.loc[df_aqi["AQI"].idxmin(), "Date"].strftime("%d %b %Y")

    st.markdown(
        f"""
- 🔴 **Worst air quality day:** {max_date} with an AQI of **{max_aqi:.0f}** (Severe).
- 🟢 **Best air quality day:** {min_date} with an AQI of **{min_aqi:.0f}** (Good).
- 📊 **Moderate** is the most frequent AQI category – meaning Hyderabad residents experienced **breathing discomfort on a majority of days**.
        """
    )


# ---------------------------------------------------------------------------
# 6. PAGE 2 – Air Quality Analysis
# ---------------------------------------------------------------------------
def show_analysis(df: pd.DataFrame) -> None:
    st.header("🔬 Air Quality Analysis")
    st.markdown("Explore **pollutant trends and correlations** for a custom date range.")

    df_aqi = df.dropna(subset=["AQI"])

    # --- Date range filter ---
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()

    col_start, col_end = st.columns(2)
    with col_start:
        start_date = st.date_input("From date", value=min_date, min_value=min_date, max_value=max_date)
    with col_end:
        end_date = st.date_input("To date", value=max_date, min_value=min_date, max_value=max_date)

    if start_date > end_date:
        st.error("⚠️ 'From date' must be before 'To date'.")
        return

    mask = (df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)
    df_range = df[mask].copy()
    df_range_aqi = df_range.dropna(subset=["AQI"])

    if df_range.empty:
        st.warning("No records found for the selected date range.")
        return

    st.info(
        f"Showing **{len(df_range):,} records** from "
        f"**{start_date.strftime('%d %b %Y')}** to **{end_date.strftime('%d %b %Y')}**."
    )

    # --- 1. Daily AQI trend over filtered range ---
    st.subheader("📈 Daily AQI Trend")
    if not df_range_aqi.empty:
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(df_range_aqi["Date"], df_range_aqi["AQI"], color="#4c9be8", linewidth=1.0)
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("AQI", fontsize=11)
        ax.set_title("Daily AQI – Selected Date Range", fontsize=13)
        fig.autofmt_xdate()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("No valid AQI data for this range.")

    # --- 2. Monthly average AQI ---
    st.subheader("📅 Monthly Average AQI")
    if not df_range_aqi.empty:
        monthly = (
            df_range_aqi.set_index("Date")["AQI"]
            .resample("MS")          # month start
            .mean()
            .dropna()
        )
        if not monthly.empty:
            fig2, ax2 = plt.subplots(figsize=(13, 4))
            ax2.bar(monthly.index, monthly.values, width=20, color="#4c9be8", alpha=0.85)
            ax2.plot(monthly.index, monthly.values, color="#ff7c40", linewidth=1.5, marker="o", markersize=4)
            ax2.set_xlabel("Month", fontsize=11)
            ax2.set_ylabel("Average AQI", fontsize=11)
            ax2.set_title("Monthly Average AQI", fontsize=13)
            fig2.autofmt_xdate()
            ax2.grid(True, alpha=0.3, axis="y")
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

    # --- 3. PM2.5 trend ---
    st.subheader("🌫️ PM2.5 Trend (µg/m³)")
    df_pm25 = df_range.dropna(subset=["PM2.5"])
    if not df_pm25.empty:
        fig3, ax3 = plt.subplots(figsize=(13, 3.5))
        ax3.plot(df_pm25["Date"], df_pm25["PM2.5"], color="#e57373", linewidth=0.9, alpha=0.85)
        ax3.set_xlabel("Date", fontsize=11)
        ax3.set_ylabel("PM2.5 (µg/m³)", fontsize=11)
        ax3.set_title("Daily PM2.5 Concentration", fontsize=13)
        fig3.autofmt_xdate()
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)
    else:
        st.info("No PM2.5 data available for this range.")

    # --- 4. PM10 trend ---
    st.subheader("💨 PM10 Trend (µg/m³)")
    df_pm10 = df_range.dropna(subset=["PM10"])
    if not df_pm10.empty:
        fig4, ax4 = plt.subplots(figsize=(13, 3.5))
        ax4.plot(df_pm10["Date"], df_pm10["PM10"], color="#ff9800", linewidth=0.9, alpha=0.85)
        ax4.set_xlabel("Date", fontsize=11)
        ax4.set_ylabel("PM10 (µg/m³)", fontsize=11)
        ax4.set_title("Daily PM10 Concentration", fontsize=13)
        fig4.autofmt_xdate()
        ax4.grid(True, alpha=0.3)
        fig4.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)
    else:
        st.info("No PM10 data available for this range.")

    # --- 5. Mean pollutant concentrations ---
    st.subheader("🧪 Mean Pollutant Concentrations")
    present_pollutants = [p for p in POLLUTANT_COLS if p in df_range.columns]
    pol_means = df_range[present_pollutants].mean().dropna()

    if not pol_means.empty:
        fig5, ax5 = plt.subplots(figsize=(11, 4))
        ax5.bar(pol_means.index, pol_means.values, color="#42a5f5", edgecolor="white", linewidth=0.5)
        ax5.set_xlabel("Pollutant", fontsize=11)
        ax5.set_ylabel("Mean Concentration", fontsize=11)
        ax5.set_title("Average Pollutant Levels (selected range)", fontsize=13)
        ax5.grid(True, alpha=0.3, axis="y")
        fig5.tight_layout()
        st.pyplot(fig5)
        plt.close(fig5)

    # --- 6. Correlation heatmap (Matplotlib imshow) ---
    st.subheader("🔗 Pollutant Correlation Heatmap")
    corr_cols = ["AQI"] + present_pollutants
    corr_df = df_range[corr_cols].dropna(how="all")

    if len(corr_df) > 10:
        corr_matrix = corr_df.corr()
        fig6, ax6 = plt.subplots(figsize=(10, 8))
        im = ax6.imshow(corr_matrix.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
        ax6.set_xticks(range(len(corr_matrix.columns)))
        ax6.set_yticks(range(len(corr_matrix.columns)))
        ax6.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=9)
        ax6.set_yticklabels(corr_matrix.columns, fontsize=9)
        ax6.set_title("Pollutant Correlation Matrix", fontsize=13)
        # Annotate cells
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix.columns)):
                ax6.text(
                    j, i, f"{corr_matrix.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="black" if abs(corr_matrix.values[i, j]) < 0.7 else "white",
                )
        fig6.tight_layout()
        st.pyplot(fig6)
        plt.close(fig6)
    else:
        st.info("Not enough data points to compute correlations for this range.")


# ---------------------------------------------------------------------------
# 7. PAGE 3 – AQI Prediction
# ---------------------------------------------------------------------------
def show_prediction(df: pd.DataFrame) -> None:
    st.header("🔮 Next-Day AQI Prediction")
    st.markdown(
        "This page uses the **trained Random Forest Regression model** to forecast "
        "the next-day AQI based on the most recent day's pollutant readings in the dataset."
    )

    # Mandatory academic disclaimer – shown before anything else
    st.warning(
        "⚠️ **Academic Disclaimer:** This prediction is based on **historical Hyderabad "
        "air quality data ending on 1 July 2020**. It is a **retrospective, next-day "
        "forecasting demonstration** built for academic evaluation and is **NOT** a "
        "live AQI reading or current real-time forecast. The model was trained on "
        "CPCB monitoring records from 2015–2020."
    )

    st.markdown("---")

    # Try to load model and run prediction
    try:
        pipeline = load_model(MODEL_PATH)
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found:\n\n`{MODEL_PATH}`\n\nPlease train the model first by running `src/train_model.py`.")
        return
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return

    try:
        result = predict_next_day_aqi(df, pipeline=pipeline)
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        return

    # --- Prediction result display ---
    pred_aqi  = result["predicted_aqi"]
    category  = result["aqi_category"]
    advisory  = result["health_advisory"]
    color_hex = result["category_color"]
    pred_date = result["target_prediction_date"]
    obs_date  = result["based_on_observation_date"]

    # Coloured category badge
    st.markdown(
        f"""
        <div style="background:{color_hex};border-radius:12px;padding:24px;text-align:center;margin-bottom:16px;">
            <p style="color:white;font-size:16px;font-weight:600;margin:0 0 8px 0;">
                Predicted AQI for {pred_date}
            </p>
            <p style="color:white;font-size:52px;font-weight:800;margin:0;">{pred_aqi}</p>
            <p style="color:white;font-size:22px;font-weight:600;margin:4px 0 0 0;">{category}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Detail columns
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 📋 Prediction Details")
        st.table(
            pd.DataFrame(
                {
                    "Field": [
                        "Observation Date (t-1)",
                        "Prediction Date (t)",
                        "Predicted AQI",
                        "AQI Category",
                    ],
                    "Value": [obs_date, pred_date, f"{pred_aqi:.2f}", category],
                }
            ).set_index("Field")
        )

    with col_b:
        st.markdown("#### 🩺 Health Advisory")
        st.markdown(
            f"""
            <div style="background:#1e2a3a;border-left:4px solid {color_hex};
                        padding:16px;border-radius:0 8px 8px 0;">
                <p style="color:#e8ecf0;font-size:15px;margin:0;">{advisory}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # --- Previous-day inputs used ---
    st.subheader("🔢 Previous-Day Inputs Used")
    lag_summary = result.get("input_lag_summary", {})
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Previous-Day AQI",   str(lag_summary.get("previous_day_aqi",  "N/A")), "AQI_lag1")
    with c2:
        metric_card("Previous-Day PM2.5", str(lag_summary.get("previous_day_pm25", "N/A")), "µg/m³")
    with c3:
        metric_card("Previous-Day PM10",  str(lag_summary.get("previous_day_pm10", "N/A")), "µg/m³")

    st.caption(f"Feature availability: {result.get('feature_availability', '')}")

    st.markdown("---")

    # --- Model performance reference ---
    st.subheader("📊 Model Performance (Test Set)")
    st.markdown(
        "These are the **evaluated metrics on 376 unseen test-set days** (June 2019 – July 2020):"
    )
    perf_df = pd.DataFrame(
        {
            "Metric": ["MAE", "RMSE", "R² Score"],
            "Random Forest": ["11.26", "13.80", "0.8494"],
            "Baseline (Mean)": ["45.80", "–", "-1.04"],
        }
    ).set_index("Metric")
    st.table(perf_df)

    # --- AQI category reference table ---
    st.subheader("📖 AQI Category Reference")
    ref_df = pd.DataFrame(
        [
            {"Category": "Good",         "AQI Range": "0–50",   "Advisory": "Minimal health impact."},
            {"Category": "Satisfactory",  "AQI Range": "51–100", "Advisory": "Minor breathing discomfort to sensitive people."},
            {"Category": "Moderate",      "AQI Range": "101–200","Advisory": "Breathing discomfort to people with asthma and heart diseases."},
            {"Category": "Poor",          "AQI Range": "201–300","Advisory": "Breathing discomfort to most people on prolonged exposure."},
            {"Category": "Very Poor",     "AQI Range": "301–400","Advisory": "Respiratory illness on prolonged exposure."},
            {"Category": "Severe",        "AQI Range": "401+",   "Advisory": "Affects healthy people; seriously impacts those with existing diseases."},
        ]
    ).set_index("Category")
    st.table(ref_df)


# ---------------------------------------------------------------------------
# 8. PAGE 4 – About AirSense
# ---------------------------------------------------------------------------
def show_about() -> None:
    st.header("ℹ️ About AirSense")

    st.markdown(
        """
### 🌿 Project Overview
**AirSense** is a data analytics and machine learning project developed as part of the
**Community Engagement Program (CEP)** at Vardhaman College of Engineering.

The project analyses **historical air quality monitoring data for Hyderabad** (Telangana, India)
collected by the **Central Pollution Control Board (CPCB)** and builds a **Random Forest
Regression model** to forecast the next day's AQI.
        """
    )

    st.markdown("---")

    st.markdown(
        """
### 🎯 Objectives
1. Load, explore, and understand the CPCB national air quality dataset.
2. Filter and clean Hyderabad-specific daily records (2015–2020).
3. Perform Exploratory Data Analysis (EDA) with visualisations.
4. Build a lag-feature dataset with no data leakage.
5. Train and evaluate a Random Forest Regression model for next-day AQI prediction.
6. Present findings via an interactive Streamlit dashboard.
        """
    )

    st.markdown("---")

    # Technology stack
    col_t, col_d = st.columns(2)

    with col_t:
        st.markdown(
            """
### 🛠️ Technologies Used
| Technology        | Purpose                              |
|-------------------|--------------------------------------|
| Python 3.14       | Core programming language            |
| NumPy             | Numerical computing                  |
| Pandas            | Data loading, cleaning, analysis     |
| Matplotlib        | Data visualisation                   |
| Scikit-learn      | Machine learning pipeline            |
| Joblib            | Model serialisation (.pkl)           |
| Jupyter Notebook  | Exploratory analysis and EDA         |
| Streamlit         | Interactive web dashboard            |
            """
        )

    with col_d:
        st.markdown(
            """
### 📂 Dataset Information
| Property            | Details                              |
|---------------------|--------------------------------------|
| Source              | CPCB (city_day.csv)                  |
| Scope               | Hyderabad, Telangana, India          |
| Date Range          | 4 Jan 2015 – 1 Jul 2020             |
| Total Records       | 2,006 daily observations             |
| Valid AQI Records   | 1,880                                |
| Features            | 16 columns (Date, AQI, 13 pollutants + bucket) |
| Licence             | Public CPCB monitoring data          |
            """
        )

    st.markdown("---")

    st.markdown(
        """
### 🤖 Machine Learning Approach

**Algorithm:** Random Forest Regression

A Random Forest is an **ensemble of decision trees**. Each tree learns patterns from
a random subset of training data and features. Their predictions are averaged to
produce a robust, low-variance final estimate.

**Why Random Forest?**
- Handles missing values gracefully (via imputation).
- Resistant to overfitting compared to a single decision tree.
- Captures non-linear relationships between pollutants and AQI.
- Provides feature importance, aiding interpretability.

**Feature Engineering (no data leakage):**
- 13 lag-1 features: each pollutant and AQI from the **previous day** (shift(1)).
- 4 calendar features: Month, Day, Day-of-week, Day-of-year of the **prediction date**.
- Current-day pollutant readings are **never** used to predict current-day AQI.

**Train / Test Split:**
| Split     | Rows | Period                        |
|-----------|------|-------------------------------|
| Training  | 1,504 | 31 Mar 2015 – 21 Jun 2019    |
| Testing   | 376   | 22 Jun 2019 – 1 Jul 2020     |
        """
    )

    st.markdown("---")

    st.markdown(
        """
### 📈 Model Performance
| Metric       | Random Forest | Baseline (Mean Predictor) |
|--------------|---------------|---------------------------|
| MAE          | **11.26**     | 45.80                     |
| RMSE         | **13.80**     | –                         |
| R² Score     | **0.8494**    | -1.04                     |

The model explains **84.94%** of the variance in next-day AQI, reducing the
mean absolute error from 45.80 (naive mean baseline) to just 11.26 AQI units.
        """
    )

    st.markdown("---")

    st.markdown(
        """
### 🗺️ Project Workflow

```
Phase 1 → Folder structure setup & initial dataset inspection
Phase 2 → Dataset understanding and Hyderabad filtering
Phase 3 → Data cleaning and preprocessing (missing values, outliers)
Phase 4 → EDA and visualisation (trend charts, heatmaps)
Phase 5 → Feature engineering, model training & evaluation
Phase 6 → Streamlit interactive web application (this dashboard)
```
        """
    )

    st.markdown("---")

    st.info(
        "📌 **Note:** All statistics, charts, and predictions in this dashboard are computed "
        "from the historical CPCB dataset (ending 1 July 2020). This is an academic project "
        "and does **not** provide live or real-time air quality information."
    )


# ---------------------------------------------------------------------------
# 9. Sidebar navigation and main router
# ---------------------------------------------------------------------------
def main() -> None:
    # Sidebar
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center;padding:10px 0 20px 0;">
                <span style="font-size:40px;">🍃</span>
                <h2 style="margin:6px 0 2px 0;color:#e8ecf0;">AirSense</h2>
                <p style="color:#7b90a0;font-size:12px;margin:0;">
                    Hyderabad AQI Analysis<br>& Prediction Dashboard
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

        page = st.radio(
            "Navigate to",
            ["🏙️ Overview", "🔬 Air Quality Analysis", "🔮 AQI Prediction", "ℹ️ About AirSense"],
            index=0,
        )

        st.markdown("---")
        st.markdown(
            "<p style='color:#7b90a0;font-size:11px;'>Data: CPCB city_day.csv<br>"
            "Period: Jan 2015 – Jul 2020<br>"
            "Model: Random Forest Regression</p>",
            unsafe_allow_html=True,
        )

    # Load data once
    try:
        df = load_dataset()
    except FileNotFoundError:
        st.error(
            f"❌ Dataset not found at:\n\n`{DATASET_PATH}`\n\n"
            "Please ensure `dataset/hyderabad_aqi_cleaned.csv` exists in the AirSense folder."
        )
        return

    # Route to selected page
    if page == "🏙️ Overview":
        show_overview(df)
    elif page == "🔬 Air Quality Analysis":
        show_analysis(df)
    elif page == "🔮 AQI Prediction":
        show_prediction(df)
    elif page == "ℹ️ About AirSense":
        show_about()


if __name__ == "__main__":
    main()
