import pytest
import pandas as pd
from src.analysis import sentiment_comparison, segment_sentiment_performance

@pytest.fixture
def sample_daily():
    return pd.DataFrame({
        'date': ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04'] * 2,
        'Account': ['A', 'B'] * 4,
        'classification': ['Fear', 'Fear', 'Greed', 'Greed'] * 2,
        'daily_pnl': [-10, -5, 20, 15, -8, -2, 25, 18],
        'win_rate': [0.4, 0.3, 0.6, 0.7, 0.35, 0.45, 0.65, 0.75],
        'avg_leverage': [3, 4, 8, 9, 2, 3, 7, 10],
        'trade_count': [5, 4, 10, 12, 6, 5, 11, 14],
        'long_short_ratio': [1.2, 1.0, 1.8, 2.0, 1.1, 0.9, 1.6, 1.9],
        'leverage_segment': ['Low'] * 8
    })

def test_sentiment_comparison(sample_daily):
    result = sentiment_comparison(sample_daily, 'daily_pnl')
    assert result['metric'] == 'daily_pnl'
    assert result['fear_mean'] < result['greed_mean']

def test_segment_sentiment_performance(sample_daily):
    result = segment_sentiment_performance(sample_daily, 'leverage_segment', 'daily_pnl')
    assert list(result.columns) == ['classification', 'leverage_segment', 'daily_pnl']
    assert len(result) == 2 