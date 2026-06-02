"""
EnhancedModel.py
Enhanced model with Random Forest, Gradient Boosting, and time series validation
"""


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_data(filename="aapl_data.csv"):
    """Load data from CSV file"""
    df = pd.read_csv(filename, index_col=0, parse_dates=True)
    print(f" Loaded {len(df)} rows from {filename}")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")
    return df

def engineer_advanced_features(df):
    """
    Enhanced feature engineering with multiple OI windows and lagged variables
    FIXED: Better NaN handling to preserve data
    """
    df = df.copy()
    
    # Keep only relevant columns
    ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[ohlcv_cols]
    
    print(f"   Starting with {len(df)} rows")
    
    # Basic returns (will create some NaN at row 0)
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Simulate bid/ask (no NaN created)
    df['simulated_bid'] = df['Low']
    df['simulated_ask'] = df['High']
    df['spread'] = df['simulated_ask'] - df['simulated_bid']
    df['midprice'] = (df['simulated_bid'] + df['simulated_ask']) / 2
    df['spread_pct'] = df['spread'] / (df['midprice'] + 1e-6)
    
    # Trade direction (no NaN created)
    df['price_vs_mid'] = df['Close'] - df['midprice']
    df['direction'] = np.where(df['price_vs_mid'] > 0, 1, 
                                np.where(df['price_vs_mid'] < 0, -1, 0))
    
    # Volume features (rolling creates NaN for first rows)
    df['log_volume'] = np.log(df['Volume'] + 1)
    df['volume_ma5'] = df['Volume'].rolling(5, min_periods=1).mean()
    df['volume_ratio'] = df['Volume'] / (df['volume_ma5'] + 1)
    
    # Volatility features (use min_periods to keep more data)
    df['volatility_5'] = df['returns'].rolling(5, min_periods=2).std()
    df['volatility_10'] = df['returns'].rolling(10, min_periods=2).std()
    df['volatility_ratio'] = df['volatility_5'] / (df['volatility_10'] + 1e-6)
    
    # Fill initial NaN volatilities with forward fill (for first few rows)
    df['volatility_5'] = df['volatility_5'].fillna(method='bfill').fillna(0.001)
    df['volatility_10'] = df['volatility_10'].fillna(method='bfill').fillna(0.001)
    df['volatility_ratio'] = df['volatility_ratio'].fillna(1.0)
    
    # Multiple order imbalance windows
    buy_volume = np.where(df['direction'] == 1, df['Volume'], 0)
    sell_volume = np.where(df['direction'] == -1, df['Volume'], 0)
    
    windows = [3, 5, 10, 15]  # periods
    for w in windows:
        buy_rolling = pd.Series(buy_volume).rolling(w, min_periods=1).sum()
        sell_rolling = pd.Series(sell_volume).rolling(w, min_periods=1).sum()
        df[f'oi_{w*5}min'] = (buy_rolling - sell_rolling) / (buy_rolling + sell_rolling + 1e-6)
    
    # Time features (no NaN)
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['intraday_position'] = df['hour'] * 60 + df['minute']
    
    # Interaction features
    df['volume_spread_interaction'] = df['log_volume'] * df['spread_pct']
    df['volatility_oi_interaction'] = df['volatility_5'] * df['oi_25min']
    
    # Target: next period return
    df['target_return'] = df['Close'].shift(-1) / df['Close'] - 1
    
    # Lagged features (use min_periods=1 to keep data, will create NaN at start)
    for lag in [1, 2, 3]:
        df[f'lag_volume_{lag}'] = df['log_volume'].shift(lag)
        df[f'lag_oi_{lag}'] = df['oi_25min'].shift(lag)
        df[f'lag_volatility_{lag}'] = df['volatility_5'].shift(lag)
        df[f'lag_return_{lag}'] = df['returns'].shift(lag)
    
    # Fill remaining NaNs with 0 or forward fill
    df = df.fillna(method='ffill').fillna(0)
    
    print(f"   After feature engineering: {len(df)} rows")
    print(f"   Total columns: {len(df.columns)}")
    
    return df

def prepare_features(df):
    """Select and prepare features for modeling"""
    # Define feature columns (only those that exist)
    potential_features = [
        'log_volume', 'volume_ratio', 'spread_pct', 
        'volatility_5', 'volatility_ratio', 'intraday_position',
        'volume_spread_interaction', 'volatility_oi_interaction',
        'lag_volume_1', 'lag_oi_1', 'lag_volatility_1', 'lag_return_1',
        'lag_volume_2', 'lag_oi_2', 'lag_volatility_2', 'lag_return_2'
    ]
    
    # Add OI windows
    oi_cols = [col for col in df.columns if 'oi_' in col and 'min' in col]
    potential_features.extend(oi_cols)
    
    # Filter to existing columns
    feature_cols = [col for col in potential_features if col in df.columns]
    
    X = df[feature_cols]
    y = df['target_return']
    
    # Remove outliers (but only if we have enough data)
    mask = abs(y - y.mean()) < 3 * y.std()
    X = X[mask]
    y = y[mask]
    
    # Also remove any rows with inf or NaN (should be 0 after ffill)
    X = X.replace([np.inf, -np.inf], 0)
    y = y.replace([np.inf, -np.inf], 0)
    
    print(f"   Features used: {len(feature_cols)}")
    print(f"   Samples after cleaning: {len(X)}")
    
    if len(X) == 0:
        raise ValueError("No samples left after cleaning! Check feature engineering.")
    
    return X, y, feature_cols

def train_enhanced_models(X, y, feature_cols):
    """
    Train multiple models with time series cross-validation
    """
    # Only use 3 splits if we have limited data
    n_splits = min(3, len(X) // 20)
    if n_splits < 2:
        n_splits = 2
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    models = {
        'Ridge (L2)': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    }
    
    results = {}
    
    print("\n" + "="*60)
    print(f"ENHANCED MODEL RESULTS ({n_splits}-fold Time Series CV)")
    print("="*60)
    
    for name, model in models.items():
        print(f"\n Training {name}...")
        
        cv_scores_rmse = []
        cv_scores_r2 = []
        cv_scores_direction = []
        
        try:
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Skip if validation set is too small
                if len(y_val) < 5:
                    continue
                
                # Scale features for Ridge
                if name == 'Ridge (L2)':
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_val = scaler.transform(X_val)
                
                model_copy = model.__class__(**model.get_params())
                model_copy.fit(X_train, y_train)
                y_pred = model_copy.predict(X_val)
                
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                r2 = r2_score(y_val, y_pred)
                direction_acc = (np.sign(y_val) == np.sign(y_pred)).mean()
                
                cv_scores_rmse.append(rmse)
                cv_scores_r2.append(r2)
                cv_scores_direction.append(direction_acc)
            
            if len(cv_scores_rmse) == 0:
                print(f"   No valid folds for {name}")
                continue
                
            results[name] = {
                'rmse_mean': np.mean(cv_scores_rmse),
                'rmse_std': np.std(cv_scores_rmse),
                'r2_mean': np.mean(cv_scores_r2),
                'r2_std': np.std(cv_scores_r2),
                'direction_mean': np.mean(cv_scores_direction),
                'direction_std': np.std(cv_scores_direction)
            }
            
            print(f"   RMSE: {results[name]['rmse_mean']:.6f} (+/- {results[name]['rmse_std']:.6f})")
            print(f"   R²:   {results[name]['r2_mean']:.4f} (+/- {results[name]['r2_std']:.4f})")
            print(f"   Directional Accuracy: {results[name]['direction_mean']:.2%} (+/- {results[name]['direction_std']:.2%})")
            
        except Exception as e:
            print(f"   Error training {name}: {str(e)[:50]}")
            continue
    
    return results

def get_feature_importance(X, y, feature_cols):
    """Train Random Forest on full dataset to get feature importance"""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    try:
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))
        
        return importance_df
    except Exception as e:
        print(f"   Could not compute feature importance: {e}")
        return pd.DataFrame()

def compare_with_baseline(enhanced_results, baseline_rmse, baseline_direction):
    """Compare enhanced model with baseline"""
    print("\n" + "="*60)
    print("MODEL COMPARISON vs BASELINE")
    print("="*60)
    
    print(f"\nBaseline Linear Regression:")
    print(f"   RMSE: {baseline_rmse:.6f}")
    print(f"   Directional Accuracy: {baseline_direction:.2%}")
    
    if not enhanced_results:
        print("\n No enhanced models completed successfully")
        return
    
    best_model = None
    best_improvement = -float('inf')
    
    for name, metrics in enhanced_results.items():
        rmse_improvement = ((baseline_rmse - metrics['rmse_mean']) / baseline_rmse * 100)
        dir_improvement = metrics['direction_mean'] - baseline_direction
        
        print(f"\n{name}:")
        print(f"   RMSE: {metrics['rmse_mean']:.6f} ({rmse_improvement:+.1f}% vs baseline)")
        print(f"   Directional Accuracy: {metrics['direction_mean']:.2%} ({dir_improvement:+.1%})")
        print(f"   R²: {metrics['r2_mean']:.4f}")
        
        if rmse_improvement > best_improvement:
            best_improvement = rmse_improvement
            best_model = name
    
    if best_model:
        print(f"\n Best Enhanced Model: {best_model}")
        print(f"   RMSE Improvement: {best_improvement:.1f}%")

if __name__ == "__main__":
    print("="*60)
    print("ENHANCED MODEL PIPELINE")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    df = load_data("aapl_data.csv")
    
    # Engineer advanced features
    print("\n2. Engineering advanced features...")
    df_features = engineer_advanced_features(df)
    
    if len(df_features) == 0:
        print(" Error: No data after feature engineering!")
        exit(1)
    
    # Prepare features
    print("\n3. Preparing features...")
    X, y, feature_cols = prepare_features(df_features)
    
    if len(X) == 0:
        print(" Error: No samples after feature preparation!")
        exit(1)
    
    # Train enhanced models
    print("\n4. Training enhanced models with time series CV...")
    enhanced_results = train_enhanced_models(X, y, feature_cols)
    
    # Feature importance
    importance_df = get_feature_importance(X, y, feature_cols)
    
    # Load baseline results for comparison
    print("\n5. Loading baseline results for comparison...")
    try:
        baseline_df = pd.read_csv("baseline_results.csv")
        baseline_rmse = baseline_df['Test_RMSE'].iloc[0]
        baseline_direction = baseline_df['Directional_Accuracy'].iloc[0]
        print(f"   Baseline RMSE: {baseline_rmse:.6f}")
        print(f"   Baseline Directional Accuracy: {baseline_direction:.2%}")
    except:
        baseline_rmse = 0.001154
        baseline_direction = 0.5032
        print("   Using default baseline values from your earlier run")
    
    compare_with_baseline(enhanced_results, baseline_rmse, baseline_direction)
    
    # Save results
    if enhanced_results:
        summary = pd.DataFrame([{
            'Model': name,
            'RMSE': metrics['rmse_mean'],
            'R2': metrics['r2_mean'],
            'Directional_Accuracy': metrics['direction_mean']
        } for name, metrics in enhanced_results.items()])
        
        summary.to_csv("enhanced_results.csv", index=False)
        print("\n Results saved to 'enhanced_results.csv'")
    
    if not importance_df.empty:
        importance_df.to_csv("feature_importance.csv", index=False)
        print(" Feature importance saved to 'feature_importance.csv'")
    
    print("\n" + "="*60)
    print(" Enhanced model complete!")
    print("="*60)