# ⚡ Energy Consumption Forecasting Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Overview

Energy consumption forecasting is an important task in energy management systems. Accurate predictions help optimize energy usage, reduce wastage, and support better decision-making.

This project uses Machine Learning techniques to predict household energy consumption based on electrical measurements and time-related features. The model is trained on real-world household power consumption data and evaluated using standard regression metrics.

---

## 🎯 Objectives

* Analyze household electricity consumption patterns.
* Perform data cleaning and preprocessing.
* Extract meaningful time-based features.
* Visualize relationships between electrical parameters.
* Train a Machine Learning model for prediction.
* Evaluate model performance using regression metrics.
* Save trained components for future use.

---
## Key Results

- R² Score: 0.999
- RMSE: 0.0362
- MAE: 0.0184
- Random Forest Regressor used
- Dataset contains over 1 million records

---
## 📊 Dataset

This project uses the **Individual Household Electric Power Consumption Dataset**.

### Dataset Features

| Feature               | Description                                  |
| --------------------- | -------------------------------------------- |
| Date                  | Date of observation                          |
| Time                  | Time of observation                          |
| Global_active_power   | Total active power consumed                  |
| Global_reactive_power | Reactive power consumed                      |
| Voltage               | Household voltage                            |
| Global_intensity      | Current intensity                            |
| Sub_metering_1        | Kitchen energy consumption                   |
| Sub_metering_2        | Laundry energy consumption                   |
| Sub_metering_3        | Water heater and air conditioner consumption |

### Dataset Source

The dataset is available from the UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

> Note: The dataset is not included in this repository because of GitHub file size limitations.
>
> Download `household_power_consumption.csv` from the UCI repository and place it in the project root before running `Energy Consumption Forecasting.py`.
>
> Dataset URL: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Joblib

---

## 🔄 Project Workflow

### 1. Data Collection

* Household power consumption dataset loaded from CSV.

### 2. Data Preprocessing

* Missing value handling
* Data type conversion
* Duplicate removal
* DateTime conversion
* Feature extraction

### 3. Feature Engineering

Additional features were created:

* Hour
* Day
* Month

### 4. Exploratory Data Analysis (EDA)

Several visualizations were created to understand the dataset:

* Power Distribution
* Voltage vs Power Consumption
* Correlation Matrix
* Actual vs Predicted Plot
* Feature Importance Analysis

### 5. Model Training

Algorithm used:

**Random Forest Regressor**

Reasons for selection:

* Handles large datasets efficiently
* Captures non-linear relationships
* Reduces overfitting
* Provides feature importance scores

### 6. Model Evaluation

Performance was measured using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

---

## 📈 Results

### Model Performance

| Metric   | Score  |
| -------- | ------ |
| MAE      | 0.0184 |
| RMSE     | 0.0362 |
| R² Score | 0.999  |

The model achieved excellent predictive performance on the test dataset.

---

## 🔍 Feature Importance

The most influential features were:

1. Global Intensity
2. Voltage
3. Global Reactive Power
4. Hour
5. Sub Metering Values

Feature importance analysis helps understand the factors affecting energy consumption.

---

## 🖼 Project Visualizations

### Power Distribution

![Power Distribution](Images/Global_Active_Power_Distirbution.png)

### Voltage vs Power Consumption

![Voltage vs Power](Images/Voltage_VS_Power_Consumtion.png)

### Correlation Matrix

![Correlation Matrix](Images/Correlation_matrix.png)

### Actual vs Predicted

![Actual vs Predicted](Images/Actual_VS_Priedicted.png)

### Feature Importance

![Feature Importance](Images/Feature_Importance.png)

---

## 📂 Repository Structure

```text
Energy-Consumption-Forecasting/
│
├── Images/
│   ├── Global_Active_Power_Distribution.png
│   ├── Voltage_VS_Power_Consumption.png
│   ├── Correlation_matrix.png
│   ├── Actual_VS_Predicted.png
│   └── Feature_Importance.png
│
├── Energy Consumption Forecasting.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/gorlejeevan44-debug/Energy_Consumption_Forecasting.git
```

Navigate into the project folder:

```bash
cd Energy_Consumption_Forecasting
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Place the downloaded dataset inside the project directory and run:

```bash
python "Energy Consumption Forecasting.py"
```

The script will:

* Load and preprocess data
* Train the Random Forest model
* Generate visualizations
* Evaluate model performance
* Save preprocessing components

---

## 📚 Skills Demonstrated

* Data Cleaning
* Data Preprocessing
* Exploratory Data Analysis
* Feature Engineering
* Data Visualization
* Machine Learning
* Regression Analysis
* Model Evaluation
* Model Persistence

---

## 🚀 Future Improvements

Potential future enhancements:

* Time Series Forecasting
* XGBoost Regression
* LSTM Neural Networks
* Streamlit Dashboard
* Real-Time Monitoring
* Cloud Deployment

---

## 👨‍💻 Author

### Gorle Jani Venkata Pavan Sai Jeevan

Engineering Student | Python Developer | Machine Learning & Data Analytics Enthusiast

Interested in building real-world Machine Learning and Data Analytics solutions using Python.

---

⭐ If you found this project useful, consider giving it a star.
