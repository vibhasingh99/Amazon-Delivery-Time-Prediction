import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

model_path = r"C:\Users\BLUEZONE\Desktop\Amazone Delivery\random_forest_delivery_model_v2.pkl"
features_path = r"C:\Users\BLUEZONE\Desktop\Amazone Delivery\model_features_v2.pkl"
model = joblib.load(model_path)
features = joblib.load(features_path)

def preprocess_data(df, features):
    categorical_cols = ['Weather','Traffic','Vehicle','Area']
    
    for col in categorical_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
    
    for col in features:
        if col not in df.columns:
            df[col] = 0
    
    df_model = df[features]
    return df_model

st.title("Amazon Delivery Time Predictor 🚚")
st.write("Upload a CSV to predict delivery times or use the sidebar for manual input.")

st.sidebar.header("Manual Order Input")
distance_km = st.sidebar.number_input("Distance (km)", 0.0, 500.0, 5.0)
pickup_delay_mins = st.sidebar.number_input("Pickup Delay (mins)", 0, 180, 10)
order_hour = st.sidebar.number_input("Order Hour (0-23)", 0, 23, 12)
order_weekday = st.sidebar.number_input("Order Weekday (0=Mon,6=Sun)", 0, 6, 2)
Weather = st.sidebar.selectbox("Weather", ["Sunny", "Rainy", "Fog", "Stormy", "Sandstorms", "Windy"])
Traffic = st.sidebar.selectbox("Traffic", ["Low", "Medium", "High", "Jam", "NaN"])
Vehicle = st.sidebar.selectbox("Vehicle", ["scooter", "van", "bike"])
Area = st.sidebar.selectbox("Area", ["Urban", "Semi-Urban", "Other"])
Agent_Rating = st.sidebar.slider("Agent Rating", 1.0, 5.0, 4.0)

manual_input = pd.DataFrame({
    'distance_km':[distance_km],
    'pickup_delay_mins':[pickup_delay_mins],
    'order_hour':[order_hour],
    'order_weekday':[order_weekday],
    'Weather':[Weather],
    'Traffic':[Traffic],
    'Vehicle':[Vehicle],
    'Area':[Area],
    'Agent_Rating':[Agent_Rating]
})

manual_processed = preprocess_data(manual_input, features)
manual_prediction = model.predict(manual_processed)[0]

st.sidebar.subheader("Predicted Delivery Time (hours)")
st.sidebar.success(f"{manual_prediction:.2f} hrs")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("CSV loaded successfully!")
        st.dataframe(df.head())

        df_model = preprocess_data(df, features)
        df['Predicted_Delivery_Time_hours'] = model.predict(df_model)

        st.subheader("Predictions")
        st.dataframe(df.head(20))
        st.subheader("Summary Statistics")
        st.write(df['Predicted_Delivery_Time_hours'].describe())

        st.subheader("Select an Order to Explore")
        order_index = st.number_input("Order index (row number)", 0, len(df)-1, 0)
        selected_order = df.iloc[[order_index]]

        st.write("### Selected Order Details")
        st.dataframe(selected_order)

        st.write("### Predicted Delivery Time")
        st.success(f"{selected_order['Predicted_Delivery_Time_hours'].values[0]:.2f} hrs")

    
        fig = px.histogram(df, x='Predicted_Delivery_Time_hours', nbins=30,
                           title="Delivery Time Distribution", marginal="box", color_discrete_sequence=['skyblue'])
        fig.add_vline(x=selected_order['Predicted_Delivery_Time_hours'].values[0],
                      line_dash="dash", line_color="red", annotation_text="Selected Order", annotation_position="top right")
        st.plotly_chart(fig)

        if 'Traffic' in df.columns:
            fig = px.box(df, x='Traffic', y='Predicted_Delivery_Time_hours', color='Traffic',
                         title="Traffic vs Predicted Delivery Time")
            fig.add_trace(go.Scatter(
                x=[selected_order['Traffic'].values[0]],
                y=[selected_order['Predicted_Delivery_Time_hours'].values[0]],
                mode='markers',
                marker=dict(color='red', size=12),
                name="Selected Order"
            ))
            st.plotly_chart(fig)

        if 'Weather' in df.columns:
            fig = px.box(df, x='Weather', y='Predicted_Delivery_Time_hours', color='Weather',
                         title="Weather vs Predicted Delivery Time")
            fig.add_trace(go.Scatter(
                x=[selected_order['Weather'].values[0]],
                y=[selected_order['Predicted_Delivery_Time_hours'].values[0]],
                mode='markers',
                marker=dict(color='red', size=12),
                name="Selected Order"
            ))
            st.plotly_chart(fig)

        if 'Agent_Rating' in df.columns:
            fig = px.scatter(df, x='Agent_Rating', y='Predicted_Delivery_Time_hours', color='Agent_Rating',
                             title="Agent Rating vs Predicted Delivery Time", hover_data=df.columns)
            fig.add_trace(go.Scatter(
                x=[selected_order['Agent_Rating'].values[0]],
                y=[selected_order['Predicted_Delivery_Time_hours'].values[0]],
                mode='markers',
                marker=dict(color='red', size=15),
                name="Selected Order"
            ))
            st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload your CSV file to get started.")
