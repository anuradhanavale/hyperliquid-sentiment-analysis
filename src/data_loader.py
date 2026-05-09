import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def load_fear_greed_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_DIR / "fear_greed_index.csv")
    df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date
    df = df[['date', 'value', 'classification']]
    return df

def load_trader_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_DIR / "historical_data.csv")
    # Timestamp column is in milliseconds (1.73E+12)
    df['timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df['date'] = df['timestamp'].dt.date
    return df

def merge_data(trader_df: pd.DataFrame, fear_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(trader_df, fear_df, on='date', how='left')
    merged['classification'] = merged['classification'].fillna(method='ffill')
    return merged

def save_processed_data(df: pd.DataFrame, filename: str = "merged_data.parquet"):
    PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
    df.to_parquet(PROCESSED_DIR / filename, index=False)

def load_processed_data(filename: str = "merged_data.parquet") -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / filename)

if __name__ == "__main__":
    fear = load_fear_greed_data()
    trader = load_trader_data()
    merged = merge_data(trader, fear)
    save_processed_data(merged)
    print(f"Merged data shape: {merged.shape}")