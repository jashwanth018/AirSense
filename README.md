# 🍃 AirSense – Hyderabad Air Quality Analysis & AQI Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12%2B-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Random%20Forest-success?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge" />
</p>

<p align="center">
  <b>Understand the air. Discover the patterns. Predict tomorrow.</b>
</p>

<p align="center">
  An interactive data analytics and machine learning application that explores Hyderabad's historical air pollution patterns and predicts next-day Air Quality Index (AQI) using Random Forest Regression.
</p>

<p align="center">
  🌐 <b>Live Application:</b> <a href="https://airsense-bfyexghtw6nqdixpttt7nt.streamlit.app/">Launch AirSense Dashboard</a>
</p>

---

## 🌍 About the Project

**AirSense** is a data analytics and machine learning project developed as part of the Community Engagement Program (CEP) at **Vardhaman College of Engineering**.

The project analyses historical air quality monitoring records for Hyderabad, Telangana, India. It explores pollutant behaviour, identifies air quality trends and builds a machine learning model to predict the following day's AQI.

The application brings together data preprocessing, exploratory data analysis, machine learning and an interactive Streamlit dashboard.

## ✨ Key Features

- 📊 **Interactive Dashboard** – Explore Hyderabad's historical air quality data.
- 🌫️ **Pollutant Analysis** – Study PM2.5, PM10, CO, NO₂, SO₂ and other pollutants.
- 📈 **Trend Visualization** – Understand daily and monthly AQI variations.
- 🔥 **Correlation Analysis** – Examine relationships between pollutants and AQI.
- 🤖 **Machine Learning Prediction** – Predict next-day AQI using Random Forest Regression.
- 🩺 **AQI Category & Advisory** – Display the predicted category and corresponding health advisory.
- 📅 **Date Range Filtering** – Analyse selected historical periods.

## 📂 Dataset Overview

| Property | Details |
|---|---|
| Dataset | CPCB city_day.csv |
| Geographic scope | Hyderabad, Telangana, India |
| Original dataset | 29,531 records |
| Cities in original dataset | 26 |
| Hyderabad records | 2,006 |
| Monitoring period | January 2015 – July 2020 |
| Valid AQI observations | 1,880 |
| Missing AQI observations | 126 |

**Data handling:** Missing pollutant readings are handled using limited past-only forward filling. Missing AQI target values are preserved and excluded from model training. Extreme observations are not automatically deleted.

## 🧠 Machine Learning Approach

AirSense uses **Random Forest Regression** to estimate the next day's AQI.

### Model workflow

1. Sort Hyderabad observations chronologically.
2. Generate previous-day pollutant and AQI lag features.
3. Extract calendar features such as month, day and day of the week.
4. Split observations chronologically into training and testing sets.
5. Handle missing predictor values using a median imputer fitted on training data.
6. Train and evaluate the Random Forest Regression model.
7. Generate next-day AQI predictions.

### Model configuration

| Parameter | Value |
|---|---|
| Algorithm | Random Forest Regressor |
| Number of estimators | 100 |
| Maximum depth | 10 |
| Random state | 42 |
| Total input features | 17 |
| Training observations | 1,504 |
| Testing observations | 376 |
| Train-test split | 80:20 |

### 📊 Model Performance

| Evaluation Metric | Test Result |
|---|---:|
| MAE | 11.26 |
| RMSE | 13.80 |
| R² Score | 0.8494 |

**Interpretation:** The model achieved an R² score of approximately 0.85 on the chronological test set.

## 📉 Exploratory Data Analysis

The project includes visualizations for:

- Daily AQI trends
- Monthly average AQI
- PM2.5 concentration trends
- PM10 concentration trends
- PM2.5 versus PM10 comparison
- Average pollutant levels
- AQI category distribution
- Pollutant correlation heatmap

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Scikit-learn | Machine learning |
| Joblib | Model serialization |
| Jupyter Notebook | Data exploration and experimentation |
| Streamlit | Interactive web application |

## 🏗️ Project Structure

```text
AirSense/
│
├── dataset/
│   ├── city_day.csv
│   ├── hyderabad_aqi_raw.csv
│   └── hyderabad_aqi_cleaned.csv
│
├── notebooks/
│   └── analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train_model.py
│   └── prediction.py
│
├── visualizations/
│   ├── aqi_trend.png
│   ├── monthly_aqi_trend.png
│   ├── pm25_trend.png
│   ├── pm10_trend.png
│   ├── pm25_vs_pm10.png
│   ├── pollutant_averages.png
│   ├── aqi_category_distribution.png
│   ├── pollutant_correlation_heatmap.png
│   ├── actual_vs_predicted_line.png
│   ├── actual_vs_predicted_scatter.png
│   └── feature_importance.png
│
├── models/
│   └── random_forest_aqi.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Run AirSense Locally

### 1. Clone the repository

```bash
git clone https://github.com/jashwanth018/AirSense.git
```

### 2. Navigate to the project directory

```bash
cd AirSense
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment (Windows)

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Launch the application

```bash
python -m streamlit run app.py
```

Open your browser at:

```text
http://localhost:8501
```

## 🔮 Sample Prediction

The model uses the latest historical observation available in the dataset to demonstrate next-day prediction.

| Property | Result |
|---|---|
| Latest observation | 01 July 2020 |
| Prediction date | 02 July 2020 |
| Predicted AQI | 67.59 |
| Predicted category | Satisfactory |

## ⚠️ Important Disclaimer

This project uses historical air quality records ending on **1 July 2020**.

The displayed AQI prediction is a retrospective academic demonstration. It is **not a live AQI measurement or a real-time forecast for 2026**.

The model's performance is based on the available historical test data and should not be interpreted as a guarantee of future prediction accuracy.

## 🎯 Project Objectives

- Analyse historical urban air pollution patterns.
- Understand the behaviour of major air pollutants.
- Apply machine learning to environmental data.
- Develop a predictive AQI model.
- Present analytical findings through an interactive dashboard.

## 👨‍💻 Developed As

**Course End Project (CEP)-BIG DATA TOOLS**  
Department of Electronics and Communication Engineering  
**Vardhaman College of Engineering**  
Hyderabad, Telangana, India.

## 🤝 Contributions

Suggestions and improvements are welcome. Feel free to explore the repository and share feedback.

## ⭐ Support

If you find this project interesting, consider giving the repository a ⭐ on GitHub!

---

<p align="center">
  <b>🍃 AirSense – Turning Historical Air Quality Data into Meaningful Insights.</b>
</p>

<p align="center">
  Built with Python, Machine Learning and Streamlit.
</p>
