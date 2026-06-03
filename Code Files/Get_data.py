"""
Get_data.py
Downloads AAPL 5-minute data from Yahoo Finance and saves to CSV
"""

import yfinance as yf
import pandas as pd
from datetime import datetime

def download_aapl_data(period="1mo", interval="5m"):
    """
    Download AAPL data from Yahoo Finance
    
    Parameters:
    - period: time period (default "1mo")
    - interval: bar size (default "5m")
    
    Returns:
    - DataFrame with OHLCV data
    """
    print(f"Downloading AAPL {interval} data for period {period}...")
    
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period=period, interval=interval)
    
    if data.empty:
        raise Exception("No data downloaded. Check your internet connection or try different parameters.")
    
    # Remove timezone info for cleaner CSV
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    
    print(f" Downloaded {len(data)} rows")
    print(f"   Date range: {data.index.min()} to {data.index.max()}")
    
    return data

def save_to_csv(data, filename="aapl_data.csv"):
    """Save DataFrame to CSV file"""
    data.to_csv(filename)
    print(f" Data saved to {filename}")
    
    # Show file info
    import os
    file_size = os.path.getsize(filename) / 1024  # KB
    print(f"   File size: {file_size:.1f} KB")

def load_from_csv(filename="aapl_data.csv"):
    """Load data from CSV file"""
    data = pd.read_csv(filename, index_col=0, parse_dates=True)
    print(f" Loaded {len(data)} rows from {filename}")
    return data

if __name__ == "__main__":
    # Run this script directly to download and save data
    print("="*50)
    print("DATA DOWNLOAD SCRIPT")
    print("="*50)
    
    # Download data
    df = download_aapl_data(period="1mo", interval="5m")
    
    # Save to CSV
    save_to_csv(df, "aapl_data.csv")
    
    # Verify
    print("\n" + "="*50)
    print("VERIFICATION")
    print("="*50)
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nLast 5 rows:")
    print(df.tail())
    
    print("\nColumn info:")
    print(df.info())