import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import compute_daily_metrics, add_trader_segments

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'date': ['2025-01-01'] * 5,
        'Account': ['A'] * 5,
        'Closed PnL': [10, -5, 20, -2, 3],
        'Size USD': [100, 200, 150, 300, 250],
        'Side': ['BUY', 'SELL', 'BUY', 'BUY', 'SELL'],
        'Leverage': [1, 5, 10, 2, 8],
        'classification': ['Greed'] * 5
    })

def test_compute_daily_metrics(sample_data):
    daily = compute_daily_metrics(sample_data)
    assert 'daily_pnl' in daily.columns
    assert 'win_rate' in daily.columns
    assert daily['daily_pnl'].iloc[0] == sum([10, -5, 20, -2, 3])

def test_add_trader_segments(sample_data):
    daily = compute_daily_metrics(sample_data)
    seg = add_trader_segments(daily)
    assert 'leverage_segment' in seg.columns
    assert 'frequency_segment' in seg.columns