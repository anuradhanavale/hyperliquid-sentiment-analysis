import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "pnl_predictor.pkl"
DAILY_METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "daily_metrics.parquet"

st.set_page_config(page_title="Predictions", layout="wide")
st.title("🔮 Next‑Day PnL Predictions (Bonus)")

# Load data
df = pd.read_parquet(DAILY_METRICS_PATH)
df['date'] = pd.to_datetime(df['date'])

# Load model if exists
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
    st.success("Model loaded successfully.")
else:
    st.warning("Model not found. Run `04_predictive_modeling.ipynb` first.")
    model = None

# User input for prediction
st.subheader("Simulate a Trader's Next Day Performance")
col1, col2 = st.columns(2)
with col1:
    leverage = st.slider("Today's avg leverage", 1.0, 20.0, 5.0)
    win_rate = st.slider("Today's win rate", 0.0, 1.0, 0.5)
with col2:
    trades = st.slider("Number of trades today", 1, 100, 20)
    sentiment = st.selectbox("Today's sentiment", ['Fear', 'Greed'])

sentiment_num = 0 if sentiment == 'Fear' else 1

if model:
    # Dummy previous PnL (in real app, fetch from last known day)
    prev_pnl = df['daily_pnl'].mean()
    input_data = pd.DataFrame([[prev_pnl, win_rate, leverage, trades, sentiment_num, leverage]],
                              columns=['prev_1_pnl', 'prev_1_win_rate', 'prev_1_leverage',
                                       'prev_1_trades', 'sentiment_numeric', 'avg_leverage'])
    pred = model.predict(input_data)[0]
    st.metric("Predicted Next Day PnL", f"${pred:.2f}")
    
    # Classification (positive/negative)
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(input_data)[0]
        st.write(f"Probability of positive PnL: {proba[1]:.2%}")

# Show recent actual performance
st.subheader("Recent Actual PnL (last 7 days)")
recent = df.groupby('date')['daily_pnl'].mean().last('7D').reset_index()
st.line_chart(recent.set_index('date'))