import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from scipy.stats import norm

st.set_page_config(page_title="AI Market Risk Engine", layout="wide")
st.title("📉 AI/ML Market Risk Management System")

# --- SIDEBAR: INTERACTIVE USER INPUTS ---
st.sidebar.header("1. Portfolio & Market Inputs")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, RELIANCE.NS)", "AAPL")
portfolio_val = st.sidebar.number_input("Portfolio Value ($)", min_value=1000, value=100000, step=5000)
confidence = st.sidebar.slider("VaR Confidence Level (%)", 90, 99, 95)

st.sidebar.header("2. Stress Test Scenario")
market_shock = st.sidebar.slider("Simulate Market Crash (%)", -50, 0, -15)
vol_multiplier = st.sidebar.slider("Volatility Spike Multiplier", 1.0, 4.0, 2.0)

# --- LOAD DATA ---
@st.cache_data
def load_data(symbol):
    df = yf.download(symbol, period="5y", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df['Return'] = df['Close'].pct_change()
    df['Volatility_21d'] = df['Return'].rolling(21).std()
    df['Vol_Change'] = df['Volume'].pct_change()
    return df.dropna()

df = load_data(ticker)

if df.empty:
    st.error("Invalid ticker or no data found.")
    st.stop()

# --- 1. ML VOLATILITY FORECASTING (Random Forest) ---
df['Target_Vol'] = df['Volatility_21d'].shift(-1)
train_df = df.dropna()

features = ['Return', 'Volatility_21d', 'Vol_Change']
X = train_df[features].replace([np.inf, -np.inf], 0)
y = train_df['Target_Vol']

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

latest_features = df[features].iloc[[-1]].replace([np.inf, -np.inf], 0)
predicted_daily_vol = rf_model.predict(latest_features)[0]
annualized_vol = predicted_daily_vol * np.sqrt(252)

# --- 2. VALUE AT RISK (VaR) CALCULATION ---
z_score = norm.ppf(1 - (confidence / 100))
var_pct = abs(z_score * predicted_daily_vol)
var_dollar = portfolio_val * var_pct

# --- 3. ANOMALY DETECTION (Isolation Forest) ---
iso = IsolationForest(contamination=0.02, random_state=42)
df['Anomaly'] = iso.fit_predict(df[['Return', 'Volatility_21d']])
anomalies = df[df['Anomaly'] == -1]

# --- 4. STRESS TEST CALCULATION ---
stressed_var = var_dollar * vol_multiplier
stressed_loss = (portfolio_val * abs(market_shock / 100)) + stressed_var

# --- DISPLAY METRICS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Predicted Annual Volatility", f"{annualized_vol:.2%}")
col2.metric(f"1-Day {confidence}% VaR (Max Loss)", f"${var_dollar:,.2f}", f"-{var_pct:.2%}")
col3.metric("Abnormal Days Detected (5Y)", f"{len(anomalies)} days")
col4.metric("Stress Test Projected Loss", f"${stressed_loss:,.2f}", f"{market_shock}% Shock")

# --- INTERACTIVE CHARTS ---
st.subheader(f"Abnormal Market Movements Detected: {ticker}")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
fig.add_trace(go.Scatter(
    x=anomalies.index, y=anomalies['Close'], mode='markers',
    marker=dict(color='red', size=8), name='ML Anomaly Flag'
))
st.plotly_chart(fig, use_container_width=True)
