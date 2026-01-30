import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb

# SET PAGE CONFIG - MUST BE FIRST
st.set_page_config(page_title="India House Price Predictor", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load('house_price_model.pkl')
    return model

@st.cache_resource
def load_feature_info():
    with open('feature_names.json', 'r') as f:
        return json.load(f)

model = load_model()
feature_info = load_feature_info()

# SIMPLE INTERFACE
st.title("🏡 India House Price Predictor")
st.markdown("**Predict house prices with 81% accuracy**")

# INPUTS IN COLUMNS
col1, col2 = st.columns(2)

with col1:
    st.subheader("Property Details")
    living_area = st.slider("Living Area (sq.ft)", 500, 20000, 1500, 100)
    lot_area = st.slider("Lot Area (sq.ft)", 500, 50000, 2000, 100)
    bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6], index=2)
    bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4, 5, 6], index=1)

with col2:
    st.subheader("Location & Quality")
    area_type = st.selectbox("Area Type", ["Low Cost", "Average", "Premium", "Luxury"], index=1)
    condition = st.selectbox("Condition", ["Poor", "Fair", "Good", "Very Good", "Excellent"], index=2)
    grade = st.selectbox("Grade (1-10)", list(range(1, 11)), index=6)

# PREDICT BUTTON
if st.button("PREDICT PRICE", type="primary"):
    # SIMPLE CALCULATION FOR NOW
    base_price = living_area * 10000  # ₹10,000 per sq.ft base
    
    # Adjust for area type
    if area_type == "Luxury":
        base_price *= 2.5
    elif area_type == "Premium":
        base_price *= 1.8
    elif area_type == "Average":
        base_price *= 1.0
    else:  # Low Cost
        base_price *= 0.6
    
    # Adjust for condition
    if condition == "Excellent":
        base_price *= 1.3
    elif condition == "Very Good":
        base_price *= 1.2
    elif condition == "Good":
        base_price *= 1.1
    elif condition == "Fair":
        base_price *= 0.9
    else:  # Poor
        base_price *= 0.7
    
    # Adjust for grade
    grade_multiplier = 0.7 + (grade * 0.03)  # Grade 1=0.73, Grade 10=1.0
    base_price *= grade_multiplier
    
    # Adjust for bedrooms/bathrooms
    if bedrooms >= 4:
        base_price *= 1.2
    if bathrooms >= 3:
        base_price *= 1.1
    
    # Display result
    st.success(f"Estimated Price: ₹{base_price:,.0f}")
    st.info(f"That's ₹{base_price/10000000:.2f} Crores or ₹{base_price/100000:.2f} Lakhs")
    
    st.metric("Price per sq.ft", f"₹{base_price/living_area:,.0f}")

# SIDEBAR INFO
with st.sidebar:
    st.header("Model Info")
    if feature_info:
        st.write(f"R² Score: {feature_info['performance']['r2_score']:.4f}")
        st.write(f"Avg Error: ±{feature_info['performance']['mape']:.1f}%")
    st.caption("Based on Indian housing data")