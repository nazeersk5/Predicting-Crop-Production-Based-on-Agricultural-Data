import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px

# === Load and process the dataset ===
def load_and_process_data(file):
    df = pd.read_excel(file)
    df.columns = [col.strip().replace(" ", "_") for col in df.columns]

    df_filtered = df[['Area', 'Item', 'Year', 'Element', 'Value']]
    df_pivot = df_filtered.pivot_table(
        index=['Area', 'Item', 'Year'],
        columns='Element',
        values='Value',
        aggfunc='sum'
    ).reset_index()
    df_pivot.columns.name = None

    df_pivot.rename(columns={
        'Area harvested': 'Area_Harvested',
        'Production': 'Production'
    }, inplace=True)

    if 'Area_Harvested' in df_pivot.columns and 'Production' in df_pivot.columns:
        df_pivot['Yield'] = df_pivot['Production'] / df_pivot['Area_Harvested']
    else:
        raise ValueError("Dataset missing Area_Harvested and Production.")

    df_model = df_pivot[['Area', 'Item', 'Area_Harvested', 'Production', 'Year', 'Yield']].dropna()
    return df_model

# === Train the model ===
def train_model(df_model):
    X = df_model[['Area_Harvested', 'Yield', 'Year']]
    y = df_model['Production']

    if not np.isfinite(X).all().all():
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.dropna()
        y = y.loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=300, max_depth=14, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    joblib.dump(model, "crop_rf_model.pkl")
    joblib.dump(scaler, "crop_scaler.pkl")

    return model, scaler, mae, mse, r2

# === Load saved model ===
def load_model_and_scaler():
    model = joblib.load("crop_rf_model.pkl")
    scaler = joblib.load("crop_scaler.pkl")
    return model, scaler

# === Streamlit App ===
st.set_page_config(page_title="Smart Crop Dashboard", layout="wide")
st.markdown("""
    <style>
        .main { background-color: #f5f7fa; color: #0a0a23; }
        h1, h2, h3, h4 { color: #3e3e66; }
        .stButton>button { background-color: #5a7d9a; color: white; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("🌾 Smart Crop Production Dashboard")

st.markdown("""
This interactive dashboard allows you to:
- 📈 Explore agricultural data trends
- 🤖 Predict crop production with business insights
- 📊 Visualize insights in a modern and interactive way
""")

uploaded_file = st.file_uploader("📤 Upload your cleaned FAOSTAT Excel file", type=["xlsx"])

if uploaded_file is not None:
    df_model = load_and_process_data(uploaded_file)

    # === Dashboard ===
    st.header("📊 Crop Insights Dashboard")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Agricultural Regions")
        region_df = df_model.groupby('Area')['Production'].mean().sort_values(ascending=False).head(10).reset_index()
        fig_area = px.bar(region_df, x='Production', y='Area', orientation='h', color='Area', title="Top Producing Regions", color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_area, use_container_width=True)

    with col2:
        st.subheader("Crop Distribution")
        crop_df = df_model['Item'].value_counts().head(10).reset_index()
        crop_df.columns = ['Crop', 'Count']
        fig_crop = px.pie(crop_df, names='Crop', values='Count', title="Top Cultivated Crops", color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_crop, use_container_width=True)

    st.subheader("📅 Yearly Trend Analysis")
    yearly_avg = df_model.groupby('Year')[['Area_Harvested', 'Production']].mean().reset_index()
    fig_yearly = px.line(yearly_avg, x='Year', y=['Area_Harvested', 'Production'], markers=True, title="Yearly Crop Trends", color_discrete_map={'Area_Harvested':'#f39c12', 'Production':'#2ecc71'})
    st.plotly_chart(fig_yearly, use_container_width=True)

    # === Model Training ===
    if st.button("🚀 Train Prediction Model"):
        model, scaler, mae, mse, r2 = train_model(df_model)
        st.success("Model trained successfully!")
        st.write(f"📈 MAE: {mae:.2f} | MSE: {mse:.2f} | R²: {r2:.4f}")

    # === Prediction ===
    st.header("🎯 Predict Crop Production & Business Use")
    crop_list = df_model['Item'].unique().tolist()
    selected_crop = st.selectbox("🌾 Select Crop Type", crop_list)

    c1, c2, c3 = st.columns(3)
    with c1:
        area_input = st.number_input("🌱 Area Harvested (ha)", min_value=1.0)
    with c2:
        yield_input = st.number_input("📉 Yield (tons/ha)", min_value=0.1)
    with c3:
        year_input = st.slider("📆 Year", int(df_model['Year'].min()), int(df_model['Year'].max()), step=1)

    if st.button("🔮 Predict"):
        try:
            model, scaler = load_model_and_scaler()
            X_input = np.array([[area_input, yield_input, year_input]])
            X_scaled = scaler.transform(X_input)
            prediction = model.predict(X_scaled)[0]
            st.success(f"🌾 Estimated Production for {selected_crop}: {prediction:,.2f} tons")

            # Business use-case insights
            price_per_ton = 150  # Placeholder value
            revenue = prediction * price_per_ton
            st.info(f"💼 Estimated Revenue: ₹{revenue:,.2f}")
            st.caption("(Assuming market price of ₹150 per ton)")

        except Exception as e:
            st.error(f"Prediction error: {e}")

else:
    st.warning("Please upload a dataset to proceed.")