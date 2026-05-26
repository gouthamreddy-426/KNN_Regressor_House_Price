import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="KNN Regression App",
    layout="centered"
)

# -----------------------------------
# LOAD CSS
# -----------------------------------

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css("style.css")

# -----------------------------------
# TITLE
# -----------------------------------

st.markdown("""
<div class="card">
    <h1>KNN Regression Application</h1>
    <p>
        Predict House Prices using K-Nearest Neighbors Regression
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------
# LOAD DATASET
# -----------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("Housing.csv")

df = load_data()

# -----------------------------------
# DATA PREVIEW
# -----------------------------------

st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Dataset Preview")

st.dataframe(df.head())

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------
# FEATURE SELECTION
# -----------------------------------

X = df[['area', 'bedrooms', 'bathrooms']]

y = df['price']

# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------
# FEATURE SCALING
# -----------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# -----------------------------------
# HYPERPARAMETERS
# -----------------------------------

st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("KNN Hyperparameters")

k_value = st.slider(
    "Select K Value",
    1,
    20,
    5
)

weight_option = st.selectbox(
    "Weight Function",
    ['uniform', 'distance']
)

metric_option = st.selectbox(
    "Distance Metric",
    ['euclidean', 'manhattan', 'minkowski']
)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------
# MODEL TRAINING
# -----------------------------------

model = KNeighborsRegressor(
    n_neighbors=k_value,
    weights=weight_option,
    metric=metric_option
)

model.fit(X_train_scaled, y_train)

# -----------------------------------
# PREDICTIONS
# -----------------------------------

y_pred = model.predict(X_test_scaled)

# -----------------------------------
# EVALUATION
# -----------------------------------

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

# -----------------------------------
# MODEL PERFORMANCE
# -----------------------------------

st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Model Performance")

c1, c2 = st.columns(2)

c1.metric("RMSE", f"{rmse:.2f}")

c2.metric("R² Score", f"{r2:.3f}")

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------
# VISUALIZATION
# -----------------------------------

st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Actual vs Predicted Prices")

fig, ax = plt.subplots()

ax.scatter(y_test, y_pred, alpha=0.7)

ax.set_xlabel("Actual Prices")

ax.set_ylabel("Predicted Prices")

st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------
# USER INPUT
# -----------------------------------

st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Predict House Price")

area = st.number_input(
    "Area",
    min_value=100,
    max_value=10000,
    value=1000
)

bedrooms = st.number_input(
    "Bedrooms",
    min_value=1,
    max_value=10,
    value=2
)

bathrooms = st.number_input(
    "Bathrooms",
    min_value=1,
    max_value=10,
    value=2
)

# -----------------------------------
# PREDICTION BUTTON
# -----------------------------------

if st.button("Predict Price"):

    features = np.array([
        [area, bedrooms, bathrooms]
    ])

    scaled_features = scaler.transform(features)

    prediction = model.predict(scaled_features)

    st.markdown(
        f'''
        <div class="prediction-box">
        Predicted Price: ₹ {prediction[0]:,.2f}
        </div>
        ''',
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)