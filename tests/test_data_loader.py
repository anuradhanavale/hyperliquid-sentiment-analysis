import pytest
from pathlib import Path
import pandas as pd
from src.data_loader import load_fear_greed_data, load_trader_data, merge_data

def test_load_fear_greed_data():
    df = load_fear_greed_data()
    assert isinstance(df, pd.DataFrame)
    assert 'date' in df.columns
    assert 'classification' in df.columns
    assert len(df) > 0

def test_load_trader_data():
    df = load_trader_data()
    assert isinstance(df, pd.DataFrame)
    assert 'date' in df.columns
    assert 'Account' in df.columns
    assert len(df) > 0

def test_merge_data():
    fear = load_fear_greed_data()
    trader = load_trader_data()
    merged = merge_data(trader, fear)
    assert 'classification' in merged.columns
    assert len(merged) == len(trader)  # left join