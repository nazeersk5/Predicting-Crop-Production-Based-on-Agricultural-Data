import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === Step 1: Load dataset ===
file_path = "cleaned_FAOSTAT_data.xlsx"
df = pd.read_excel(file_path)
print(f"✅ Dataset loaded successfully from {file_path}")

# === Step 2: Clean column names ===
df.columns = [col.strip().replace(" ", "_") for col in df.columns]
print(f"✅ Columns after cleaning: {df.columns.tolist()}")

# === Step 3: Pivot to get Area_Harvested and Production ===
df_filtered = df[['Area', 'Item', 'Year', 'Element', 'Value']]
df_pivot = df_filtered.pivot_table(
    index=['Area', 'Item', 'Year'],
    columns='Element',
    values='Value',
    aggfunc='sum'
).reset_index()

# Remove multi-index name
df_pivot.columns.name = None

# Rename columns for consistency
df_pivot.rename(columns={
    'Area harvested': 'Area_Harvested',
    'Production': 'Production'
}, inplace=True)

# === Step 4: Calculate Yield ===
if 'Area_Harvested' in df_pivot.columns and 'Production' in df_pivot.columns:
    df_pivot['Yield'] = df_pivot['Production'] / df_pivot['Area_Harvested']
    print("✅ Yield calculated.")
else:
    raise ValueError("❌ Dataset does not contain Area_Harvested and Production after pivoting.")

# === Step 5: Drop missing values ===
df_model = df_pivot[['Area_Harvested', 'Production', 'Year', 'Yield']].dropna()
print(f"✅ Data ready for modeling: {df_model.shape[0]} rows")

# === Step 6: Define features and target ===
X = df_model[['Area_Harvested', 'Yield', 'Year']]
y = df_model['Production']

# === Step 7: Handle NaNs and Infs ===
if not np.isfinite(X).all().all():
    print("⚠️ Detected NaN or infinite values in features. Cleaning now...")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna()
    y = y.loc[X.index]  # Keep target in sync with features

# === Step 8: Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ Train-test split done")

# === Step 9: Scale features ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Features scaled.")

# === Step 10: Train model ===
model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
model.fit(X_train_scaled, y_train)
print("✅ Model trained successfully")

# === Step 11: Save model & scaler ===
joblib.dump(model, "crop_rf_model.pkl")
joblib.dump(scaler, "crop_scaler.pkl")
print("✅ Model and scaler saved")

# === Step 12: Evaluate model ===
y_pred = model.predict(X_test_scaled)
print(f"📊 MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"📊 MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"📊 R2 Score: {r2_score(y_test, y_pred):.4f}")