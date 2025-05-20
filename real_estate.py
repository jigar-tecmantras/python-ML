import streamlit as st
import pandas as pd
import re #Regular expressions :- get number or pattern from string
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Step 1: Load and train model once (or load pretrained model)
@st.cache_data
def load_data_and_train():
    df = pd.read_csv("ahmedabad.csv")
    df.columns = df.columns.str.strip()

    # Extract BHK from Title (e.g., "2 BHK Apartment")
    def extract_bhk(title):
        match = re.search(r'(\d+)\s*BHK', str(title))
    
        return int(match.group(1)) if match else None

    df["BHK"] = df["Title"].apply(extract_bhk)

    def extract_area(value):
        match = re.search(r'(\d+(\.\d+)?)', str(value))
        return float(match.group(1)) if match else None

    def clean_price(p):
        if isinstance(p, str):
            p = p.replace("â‚¹", "").replace("₹", "").replace(",", "").strip()
            match = re.search(r'[\d.]+', p)
            if match:
                num = float(match.group())
                if 'Lac' in p:
                    return num * 100000
                elif 'Cr' in p:
                    return num * 10000000
                else:
                    return num
        return None

    def clean_price_sqft(p):
        if isinstance(p, str):
            p = p.replace("â‚¹", "").replace("₹", "").replace(",", "").strip()
            match = re.search(r'[\d.]+', p)
            if match:
                return float(match.group())
        return None

    df['value_area'] = df['value_area'].apply(extract_area)
    df['floor'] = pd.to_numeric(df['floor'], errors='coerce')
    df['price'] = df['price'].apply(clean_price)
    df['price_sqft'] = df['price_sqft'].apply(clean_price_sqft)

    df.dropna(subset=['value_area', 'floor', 'price', 'price_sqft', 'BHK'], inplace=True)

    # Drop unused columns
    df.drop(columns=["Title", "description"], inplace=True, errors="ignore")

    # One-hot encoding for categorical features if needed
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df.drop("price", axis=1)
    y = df["price"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, scaler, X.columns

model, scaler, feature_columns = load_data_and_train()

st.title("🏡 House Price Prediction - Ahmedabad")

# Step 2: Manual input UI for required features
value_area = st.number_input("Enter Area (sq. yard)", min_value=50.0, max_value=10000.0, value=100.0)
floor = st.number_input("Enter Floor Number", min_value=0, max_value=30, value=0)
price_sqft = st.number_input("Enter Price per Sqft (₹)", min_value=1000.0, max_value=20000.0, value=3000.0)
bhk = st.selectbox("Select BHK (Number of Rooms)", [1, 2, 3, 4, 5], index=1)

# Prepare input for model
input_dict = {
    "value_area": value_area,
    "floor": floor,
    "price_sqft": price_sqft,
    "BHK": bhk,
}

# Fill missing features as 0
for col in feature_columns:
    if col not in input_dict:
        input_dict[col] = 0

# Create DataFrame for input
input_df = pd.DataFrame([input_dict], columns=feature_columns)

# Scale input
input_scaled = scaler.transform(input_df)

# Predict on button click
if st.button("🔍 Predict Price"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"🏠 Predicted House Price: ₹{prediction:,.2f}")
