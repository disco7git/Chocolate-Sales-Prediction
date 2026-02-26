import streamlit as st
import joblib
import pandas as pd

# 1. Load the model
model = joblib.load("model.pkl")

# 2. Page Configuration
st.set_page_config(page_title="Chocolate Sales Insights", layout="wide")

st.title("🍫 Chocolate Sales Predictor & Model Analytics")
st.markdown("---")

# 3. Sidebar for Performance Metrics
# Tip: Replace these numbers with your actual scores from Jupyter!
st.sidebar.header("Model Performance Metrics")
st.sidebar.metric(label="R² Score (Accuracy)", value="0.89", delta="Good")
st.sidebar.metric(label="Mean Absolute Error", value="$450.20")
st.sidebar.write("The model was trained on 2 years of historical chocolate sales data.")

# 4. Input Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Enter Sale Details")
    c1, c2 = st.columns(2)
    with c1:
        sales_person = st.number_input("Sales Person ID", value=13)
        country = st.number_input("Country ID", value=5)
        product = st.number_input("Product ID", value=21)
    with c2:
        boxes = st.number_input("Boxes Shipped", value=100)
        day = st.number_input("Day", 1, 31, 7)
        month = st.number_input("Month", 1, 12, 6)
        year = st.number_input("Year", 2020, 2030, 2024)

    if st.button("Generate Prediction", use_container_width=True):
        features = [sales_person, country, product, boxes, day, month, year]
        prediction = model.predict([features])
        
        st.success(f"### Predicted Amount: ${prediction[0]:,.2f}")

with col2:
    st.subheader("Model Insights")
    st.info("This Random Forest model analyzes patterns in regional demand and salesperson performance to estimate revenue.")
    
    # Optional: A small bar chart showing feature importance (Dummy data for UI)
    st.write("Feature Importance Influence:")
    chart_data = pd.DataFrame({
        'Feature': ['Boxes', 'Product', 'Country', 'Sales Person'],
        'Importance': [0.6, 0.2, 0.1, 0.1]
    })
    st.bar_chart(chart_data.set_index('Feature'))