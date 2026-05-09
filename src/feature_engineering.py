import pandas as pd
import numpy as np
from pathlib import Path
from src.data_loader import load_processed_data

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def compute_daily_metrics(merged_df: pd.DataFrame) -> pd.DataFrame:
    df = merged_df.copy()
    if 'Leverage' not in df.columns:
        df['Leverage'] = 1.0

    daily = df.groupby(['date', 'Account']).agg(
        daily_pnl=('Closed PnL', 'sum'),
        win_rate=('Closed PnL', lambda x: (x > 0).mean()),
        trade_count=('Closed PnL', 'count'),
        avg_trade_size_usd=('Size USD', 'mean'),
        avg_leverage=('Leverage', 'mean'),
        long_short_ratio=('Side', lambda x: (x == 'BUY').sum() / max((x == 'SELL').sum(), 1))
    ).reset_index()

    # Daily drawdown proxy
    def compute_daily_drawdown(group):
        cumsum = group['Closed PnL'].cumsum()
        return cumsum.min() if len(cumsum) > 0 else 0.0
    drawdown = df.groupby(['date', 'Account']).apply(compute_daily_drawdown).reset_index(name='max_daily_drawdown')
    daily = pd.merge(daily, drawdown, on=['date', 'Account'], how='left')

    class_map = df.groupby('date')['classification'].first().to_dict()
    daily['classification'] = daily['date'].map(class_map)
    return daily

def add_trader_segments(daily_metrics: pd.DataFrame) -> pd.DataFrame:
    trader_agg = daily_metrics.groupby('Account').agg(
        avg_leverage=('avg_leverage', 'mean'),
        avg_daily_trades=('trade_count', 'mean'),
        overall_win_rate=('win_rate', 'mean')
    ).reset_index()

    trader_agg['leverage_segment'] = pd.cut(
        trader_agg['avg_leverage'],
        bins=[0, 5, 10, 100],
        labels=['Low (0-5x)', 'Medium (5-10x)', 'High (10x+)']
    )
    trader_agg['frequency_segment'] = pd.qcut(
        trader_agg['avg_daily_trades'],
        q=3,
        labels=['Low freq', 'Medium freq', 'High freq']
    )
    trader_agg['consistency_segment'] = pd.cut(
        trader_agg['overall_win_rate'],
        bins=[0, 0.4, 0.6, 1.0],
        labels=['Low win rate (<40%)', 'Medium win rate (40-60%)', 'High win rate (>60%)']
    )

    daily_with_segments = daily_metrics.merge(
        trader_agg[['Account', 'leverage_segment', 'frequency_segment', 'consistency_segment']],
        on='Account', how='left'
    )
    return daily_with_segments

if __name__ == "__main__":
    merged = load_processed_data()
    daily = compute_daily_metrics(merged)
    daily_seg = add_trader_segments(daily)
    daily_seg.to_parquet(PROCESSED_DIR / "daily_metrics.parquet", index=False)
    print("Daily metrics saved.")