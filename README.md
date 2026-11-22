# AirSense – Hyderabad Air Quality Analysis and AQI Prediction

A Python-based data analytics and machine learning project focused on analyzing daily historical air pollution patterns and predicting the Air Quality Index (AQI) for the city of **Hyderabad, India**.

---

## 📌 Project Overview

Air pollution is a pressing environmental and public health issue. **AirSense** analyzes daily ambient air quality records in Hyderabad from 2015 to 2020 and utilizes a **Random Forest Regression** model to forecast AQI levels.

The project is structured to guide learners and practitioners through the complete end-to-end data science lifecycle:
1. **Dataset Inspection & Ingestion**: Loading and understanding environmental monitoring data.
2. **Data Cleaning & Imputation**: Handling missing pollutant values systematically without silent zero-fills.
3. **Exploratory Data Analysis (EDA)**: Investigating temporal trends, seasonal patterns, and major pollutants like $\text{PM}_{2.5}$ and $\text{PM}_{10}$.
4. **Machine Learning Model**: Training a leakage-free chronological Random Forest Regressor for next-day AQI prediction.
5. **Interactive Dashboard**: Presenting real-time insights and predictions via a modern Streamlit web application.

---

## 🛠️ Technology Stack

- **Programming Language:** Python 3.10+
- **Data Manipulation & Computation:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn (Random Forest Regression)
- **Interactive Prototyping:** Jupyter Notebook / JupyterLab
- **Web Dashboard:** Streamlit

---

## 📁 Project Folder Structure

```text
AirSense/
│
├── dataset/
│   └── city_day.csv          # Raw daily air quality dataset (CPCB India)
│
├── notebooks/
│   └── analysis.ipynb        # Step-by-step EDA, data cleaning & model development
│
├── src/
│   └── prediction.py         # Modular Python scripts for ML preprocessing & inference
│
├── visualizations/           # Exported plots and charts (EDA & evaluation)
│
├── models/                   # Serialized trained model files (e.g. .pkl / .joblib)
│
├── app.py                    # Streamlit web application dashboard
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

### Purpose of Each Directory and File

- `dataset/`: Contains the raw CSV data (`city_day.csv`) containing historical daily air quality observations.
- `notebooks/`: Contains `analysis.ipynb` for interactive data exploration, cleaning, visualization, and experimental model validation.
- `src/`: Houses reusable, modular Python source code including data preparation and prediction logic (`prediction.py`).
- `visualizations/`: Dedicated folder to store exported high-resolution charts, correlation heatmaps, and trend graphs.
- `models/`: Stores exported machine learning models and scaler/preprocessor artifacts for deployment in the web app.
- `app.py`: The entry point for the Streamlit dashboard providing a beginner-friendly UI for exploring Hyderabad AQI data and predicting future air quality.
- `requirements.txt`: Pinpoints required libraries and their compatible versions.
- `README.md`: Provides complete project documentation and usage instructions.

---

## 📊 Dataset Information

- **Source Dataset:** `city_day.csv` (CPCB National Ambient Air Quality Monitoring)
- **Target City:** Hyderabad
- **Total Records for Hyderabad:** 2,006 daily records
- **Temporal Coverage:** January 4, 2015 to July 1, 2020 (~5.5 years of continuous daily data)
- **Key Features:**
  - `PM2.5`, `PM10` (Particulate Matter in $\mu\text{g/m}^3$)
  - `NO`, `NO2`, `NOx`, `NH3` (Nitrogen compounds & Ammonia)
  - `CO` (Carbon Monoxide in $\text{mg/m}^3$)
  - `SO2` (Sulfur Dioxide in $\mu\text{g/m}^3$)
  - `O3` (Ozone in $\mu\text{g/m}^3$)
  - `Benzene`, `Toluene`, `Xylene` (Volatile Organic Compounds)
  - `AQI` (Air Quality Index - Continuous Target Variable)
  - `AQI_Bucket` (Air Quality Category: Good, Satisfactory, Moderate, Poor, Very Poor, Severe)

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure Python is installed on your system. Verify with:
```bash
python --version
```

### 2. Install Dependencies
Navigate into the `AirSense` directory and install the required packages:
```bash
cd AirSense
pip install -r requirements.txt
```

### 3. Launch the Jupyter Notebook
To explore the dataset and run the analysis step by step:
```bash
jupyter lab notebooks/analysis.ipynb
# or
jupyter notebook notebooks/analysis.ipynb
```

### 4. Run the Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 📈 Project Status & Milestones

- [x] **Phase 1: Project Setup & Dataset Inspection**
  - Directory structure initialized.
  - Original dataset preserved; `city_day.csv` safely staged in `AirSense/dataset/`.
  - Initial dataset inspection conducted; 2,006 Hyderabad records identified.
- [x] **Phase 2: Dataset Understanding & Quality Analysis**
  - Date conversion, chronological sorting, and continuity verification (2,006 consecutive daily records).
  - Feature suitability and extreme value review.
  - Intermediate raw subset saved to `dataset/hyderabad_aqi_raw.csv`.
- [x] **Phase 3: Data Cleaning & Preprocessing**
  - Modular cleaning functions implemented in `src/preprocessing.py`.
  - Causality-preserving forward filling (`limit=3`) applied without zero-filling or backward leakage.
  - AQI target preserved as `NaN`; IQR outlier audit completed without artificial truncation.
  - Cleaned dataset saved to `dataset/hyderabad_aqi_cleaned.csv`.
- [x] **Phase 4: Exploratory Data Analysis (EDA) & Visualizations**
  - Statistical analysis of AQI, PM2.5, PM10, yearly and seasonal monthly breakdowns.
  - Generated and saved 8 high-resolution Matplotlib graphs in `visualizations/`.
  - Correlation analysis establishing PM2.5 ($r=0.80$) and PM10 ($r=0.66$) as primary AQI drivers.
  - Pure Matplotlib heatmap without Seaborn dependency.
- [x] **Phase 5: Chronological Random Forest Model Training & Evaluation**
  - Engineered 13 previous-day lag-1 features and 4 calendar temporal features.
  - Strict chronological train-test split (80% train: 1,504 days; 20% test: 376 days).
  - Built Scikit-Learn Pipeline combining training-median imputer and Random Forest Regressor.
  - Achieved strong test performance: $\text{MAE} = 11.26$, $\text{RMSE} = 13.80$, $R^2 = 0.8494$ (substantially outperforming baseline $\text{MAE} = 45.80$).
  - Identified $\text{PM}_{2.5}(t-1)$ (55.9%) and $\text{AQI}(t-1)$ (24.8%) as primary predictive features.
  - Exported trained pipeline to `models/random_forest_aqi.pkl`.
  - Created reusable modules `src/train_model.py` and `src/prediction.py`.
  - Generated Matplotlib evaluation graphs in `visualizations/`.
- [x] **Phase 6: Streamlit Interactive Web Application**
  - Built a complete, professional 4-page Streamlit dashboard in `app.py`.
  - **Overview page:** 5 live KPI cards (monitoring days, valid AQI days, mean/max/min AQI, top category), daily AQI line chart, AQI category bar chart, and 3 data-driven insights.
  - **Air Quality Analysis page:** Interactive date-range filter with 6 live Matplotlib charts — daily AQI trend, monthly average AQI, PM2.5 trend, PM10 trend, mean pollutant bar chart, and correlation heatmap (pure `imshow`, no Seaborn).
  - **AQI Prediction page:** Loads `models/random_forest_aqi.pkl`, calls `predict_next_day_aqi()` from `src/prediction.py`, displays colour-coded AQI badge, health advisory, previous-day lag inputs, model performance metrics, AQI reference table, and mandatory academic disclaimer.
  - **About AirSense page:** Project overview, objectives, technology stack table, dataset info, Random Forest explanation, train/test split, model metrics, and 6-phase workflow.
  - Installed `streamlit 1.67.0`; all imports, dataset loading, model loading, and prediction verified error-free.
  - `requirements.txt` updated: removed `seaborn`, added `joblib`.

---

## 🌐 Running the Dashboard

### Launch the Streamlit Web Application

```bash
cd AirSense
python -m streamlit run app.py
```

The app will open automatically at **http://localhost:8501** in your browser.

### Dashboard Sections

| Section | Description |
|---------|-------------|
| 🏙️ **Overview** | Key statistics and AQI trend at a glance |
| 🔬 **Air Quality Analysis** | Date-filtered trend charts and pollutant breakdowns |
| 🔮 **AQI Prediction** | Next-day AQI forecast using the trained Random Forest model |
| ℹ️ **About AirSense** | Project info, methodology, and model performance |

> **Note:** The prediction is based on historical CPCB data ending **1 July 2020**.
> It is an academic retrospective demonstration and is **not** a live real-time forecast.
