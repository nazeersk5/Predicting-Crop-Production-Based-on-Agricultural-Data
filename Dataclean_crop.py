# Dataclean_crop.py

import pandas as pd
import numpy as np
import os

def clean_excel_file(file_path):
    # Extract base name and generate cleaned file name
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    cleaned_file_name = f"cleaned_{file_name}.xlsx"

    print(f"\n📂 Loading: {file_path}")
    df = pd.read_excel(file_path)

    # --- Step 1: Remove fully empty rows and columns ---
    df.dropna(axis=0, how='all', inplace=True)  # Remove fully empty rows
    df.dropna(axis=1, how='all', inplace=True)  # Remove fully empty columns

    print("\n🧽 Cleaning missing values...")

    # --- Step 2: Handle missing values smartly ---
    for column in df.columns:
        missing_count = df[column].isnull().sum()
        if missing_count > 0:
            print(f"🔍 Filling {missing_count} missing values in column: {column}")

            if df[column].dtype in [np.float64, np.int64]:
                col_non_na = df[column].dropna()

                if len(col_non_na) >= 10:
                    Q1 = col_non_na.quantile(0.25)
                    Q3 = col_non_na.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    col_filtered = col_non_na[(col_non_na >= lower_bound) & (col_non_na <= upper_bound)]
                    fill_value = col_filtered.median()
                else:
                    fill_value = col_non_na.mean()

                # Replace missing values safely
                df[column] = df[column].fillna(fill_value)

            else:
                # For categorical/textual data, fill with mode or "Unknown"
                fill_value = df[column].mode()[0] if not df[column].mode().empty else "Unknown"
                df[column] = df[column].fillna(fill_value)

    # --- Step 3: Save cleaned data ---
    df.to_excel(cleaned_file_name, index=False)
    print(f"\n✅ Cleaned file saved as: {cleaned_file_name}")


# --- Entry point ---
if __name__ == "__main__":
    print("📁 Starting Excel Cleaning Process...")

    # 🔽🔽🔽 PASTE YOUR FILE PATH BELOW 🔽🔽🔽
    file_path = "FAOSTAT_data.xlsx"  # <-- Change this path to your file

    if os.path.exists(file_path):
        clean_excel_file(file_path)
    else:
        print(f"❌ File not found at: {file_path}")
