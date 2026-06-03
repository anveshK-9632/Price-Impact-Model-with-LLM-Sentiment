"""
BaseLineModel.py
Loads data from CSV, engineers features, and implements baseline linear regression
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_data(filename="aapl_data.csv"):
    """Load data from CSV file"""
    df = pd.read_csv(filename, index_col=0, parse_dates=True)
    print(f" Loaded {len(df)} rows from {filename}")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")
    return df

def engineer_features(df):
    """
    Engineer features for price impact model
    This is the baseline feature set
    """
    df = df.copy()
    
    # Keep only relevant columns (ignore Dividends and Stock Splits)
    ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[ohlcv_cols]
    
    # Simulate bid/ask from OHLC
    df['simulated_bid'] = df['Low']
    df['simulated_ask'] = df['High']
    df['spread'] = df['simulated_ask'] - df['simulated_bid']
    df['midprice'] = (df['simulated_bid'] + df['simulated_ask']) / 2
    
    # Trade direction classification (simplified Lee-Ready)
    df['price_vs_mid'] = df['Close'].values - df['midprice'].values
    df['direction'] = np.where(df['price_vs_mid'] > 0, 'BUY', 
                                np.where(df['price_vs_mid'] < 0, 'SELL', 'NEUTRAL'))
    
    # Volume features
    df['log_volume'] = np.log(df['Volume'] + 1)
    df['volume_zscore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / (df['Volume'].rolling(20).std() + 1e-6)
    
    # Volatility features
    df['rolling_volatility'] = df['Close'].pct_change().rolling(5).std()
    df['rolling_volatility'] = df['rolling_volatility'].fillna(method='bfill')
    
    # Spread features
    df['spread_pct'] = df['spread'] / (df['midprice'] + 1e-6)
    
    # Order imbalance (5-period rolling = 25 minutes)
    buy_volume = np.where(df['direction'] == 'BUY', df['Volume'].values, 0)
    sell_volume = np.where(df['direction'] == 'SELL', df['Volume'].values, 0)
    
    buy_rolling = pd.Series(buy_volume).rolling(5).sum().values
    sell_rolling = pd.Series(sell_volume).rolling(5).sum().values
    
    df['oi_5'] = (buy_rolling - sell_rolling) / (buy_rolling + sell_rolling + 1e-6)
    
    # Target: next period return (price impact)
    df['target_return'] = df['Close'].pct_change().shift(-1).values
    
    # Drop NaN values
    df = df.dropna()
    
    print(f"   Features engineered: {len(df.columns)} columns")
    print(f"   Samples after cleaning: {len(df)}")
    
    return df

def train_baseline_model(df):
    """
    Train linear regression baseline model
    """
    # Define features
    feature_cols = ['log_volume', 'volume_zscore', 'rolling_volatility', 'spread_pct', 'oi_5', 'spread']
    
    X = df[feature_cols]
    y = df['target_return']
    
    # Remove extreme outliers (beyond 3 standard deviations)
    mask = abs(y - y.mean()) < 3 * y.std()
    X = X[mask]
    y = y[mask]
    
    print(f"   After outlier removal: {len(X)} samples")
    
    # Train/test split (keep temporal order with shuffle=False)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"   Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Directional accuracy
    train_direction = (np.sign(y_train) == np.sign(y_pred_train)).mean()
    test_direction = (np.sign(y_test) == np.sign(y_pred_test)).mean()
    
    # Feature importance (coefficients)
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': model.coef_
    })
    
    return {
        'model': model,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_direction': train_direction,
        'test_direction': test_direction,
        'feature_importance': feature_importance,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred_test': y_pred_test
    }

def print_results(results):
    """Print model results in a clean format"""
    print("\n" + "="*60)
    print("BASELINE LINEAR REGRESSION RESULTS")
    print("="*60)
    
    print(f"\n Performance Metrics:")
    print(f"   Train RMSE: {results['train_rmse']:.6f}")
    print(f"   Test RMSE:  {results['test_rmse']:.6f}")
    print(f"   Train R²:   {results['train_r2']:.4f}")
    print(f"   Test R²:    {results['test_r2']:.4f}")
    
    print(f"\n🎯 Directional Accuracy (sign prediction):")
    print(f"   Train: {results['train_direction']:.2%}")
    print(f"   Test:  {results['test_direction']:.2%}")
    
    print(f"\n Feature Coefficients:")
    print(results['feature_importance'].to_string(index=False))
    
    # Interpretation
    print(f"\n Interpretation:")
    if results['test_r2'] < 0:
        print(f"    Negative R² suggests linear model is too simple")
        print(f"   → Try EnhancedModel.py for non-linear models")
    else:
        print(f"    Model explains {results['test_r2']:.1%} of variance")
    
    if results['test_direction'] > 0.55:
        print(f"    Directional accuracy > 55% is good for noisy data")
    elif results['test_direction'] > 0.52:
        print(f"    Directional accuracy slightly above random (50%)")
    else:
        print(f"    Directional accuracy near 50% = random guessing")

if __name__ == "__main__":
    print("="*60)
    print("BASELINE MODEL PIPELINE")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    df = load_data("aapl_data.csv")
    
    # Engineer features
    print("\n2. Engineering features...")
    df_features = engineer_features(df)
    
    # Train model
    print("\n3. Training baseline model...")
    results = train_baseline_model(df_features)
    
    # Print results
    print_results(results)
    
    # Save results summary
    summary = pd.DataFrame([{
        'Model': 'Linear Regression (Baseline)',
        'Test_RMSE': results['test_rmse'],
        'Test_R2': results['test_r2'],
        'Directional_Accuracy': results['test_direction']
    }])
    summary.to_csv("baseline_results.csv", index=False)
    print("\n Results saved to 'baseline_results.csv'")
    
    print("\n Baseline model complete!")