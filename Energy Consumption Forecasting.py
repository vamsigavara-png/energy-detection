import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==================================================
# LOAD DATASET
# ==================================================

DATA_FILE = "household_power_consumption.csv"
print(f"Checking for dataset file: {DATA_FILE}")

if not os.path.exists(DATA_FILE):
    print("\nERROR: Dataset file not found.")
    print("Download 'household_power_consumption.csv' from the UCI repository and place it in the project root.")
    print("Dataset URL: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption")
    sys.exit(1)

print("Loading Dataset...")

df = pd.read_csv(
    DATA_FILE,
    sep=",",
    low_memory=False
)

print(df.head())

# ==================================================
# CONVERT NUMERIC COLUMNS
# ==================================================

numeric_cols = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# ==================================================
# DATE TIME PROCESSING
# ==================================================

df["Datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    format="%d/%m/%Y %H:%M:%S",
    errors="coerce"
)

df["Hour"] = df["Datetime"].dt.hour
df["Day"] = df["Datetime"].dt.day
df["Month"] = df["Datetime"].dt.month

# ==================================================
# DATA CLEANING
# ==================================================

print("\nMissing Values")
print(df.isnull().sum())

df.dropna(inplace=True)

df.drop_duplicates(inplace=True)

print("\nDataset Shape After Cleaning")
print(df.shape)

# ==================================================
# SAMPLE DATA
# ==================================================
# Large dataset optimization

df = df.sample(
    n=100000,
    random_state=42
)

# ==================================================
# EDA
# ==================================================

plt.figure(figsize=(8,5))

sns.histplot(
    df["Global_active_power"],
    bins=30,
    kde=True
)

plt.title("Global Active Power Distribution")
plt.show()

# ==================================================

plt.figure(figsize=(8,5))

sns.scatterplot(
    x="Voltage",
    y="Global_active_power",
    data=df.sample(5000)
)

plt.title("Voltage vs Power Consumption")
plt.show()

# ==================================================

plt.figure(figsize=(10,6))

sns.heatmap(
    df[
        numeric_cols
    ].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Matrix")
plt.show()

# ==================================================
# FEATURE SELECTION
# ==================================================

X = df[
    [
        "Voltage",
        "Global_intensity",
        "Global_reactive_power",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
        "Hour",
        "Day",
        "Month"
    ]
]

y = df["Global_active_power"]

# ==================================================
# TRAIN TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==================================================
# FEATURE SCALING
# ==================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==================================================
# MODEL TRAINING
# ==================================================

print("\nTraining Model...")

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

# ==================================================
# PREDICTION
# ==================================================

y_pred = model.predict(X_test)

# ==================================================
# EVALUATION
# ==================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)

print("\nModel Performance")
print("-"*40)

print("MAE :", round(mae,4))
print("RMSE :", round(rmse,4))
print("R2 Score :", round(r2,4))

# ==================================================
# ACTUAL VS PREDICTED
# ==================================================

plt.figure(figsize=(8,5))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

plt.xlabel("Actual")
plt.ylabel("Predicted")

plt.title(
    "Actual vs Predicted"
)

plt.show()

# ==================================================
# FEATURE IMPORTANCE
# ==================================================

importance = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }
)

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")
print(importance)

plt.figure(figsize=(8,5))

sns.barplot(
    x="Importance",
    y="Feature",
    data=importance
)

plt.title(
    "Feature Importance"
)

plt.show()

# ==================================================
# SAVE MODEL
# ==================================================

joblib.dump(
    model,
    "energy_model.pkl"
)

joblib.dump(
    scaler,
    "scaler.pkl"
)

print("\nModel Saved Successfully")

# ==================================================
# SAMPLE PREDICTION
# ==================================================

sample = pd.DataFrame(
    {
        "Voltage":[235],
        "Global_intensity":[18],
        "Global_reactive_power":[0.4],
        "Sub_metering_1":[0],
        "Sub_metering_2":[1],
        "Sub_metering_3":[17],
        "Hour":[17],
        "Day":[16],
        "Month":[12]
    }
)

sample_scaled = scaler.transform(sample)

prediction = model.predict(
    sample_scaled
)

print(
    f"\nPredicted Power Consumption: {prediction[0]:.3f} kW"
)