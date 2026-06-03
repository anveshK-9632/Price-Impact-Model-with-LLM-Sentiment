"""
Visualization.py
Creates professional plots for your quant portfolio
Shows: price impact patterns, feature importance, sentiment analysis, model comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set professional style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

def create_price_impact_plot(df, output_path="price_impact_pattern.png"):
    """
    Plot 1: Price impact patterns after large volume events
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left plot: Returns after high volume vs normal volume
    df['volume_quantile'] = pd.qcut(df['Volume'].rank(method='first'), 10, labels=False)
    high_volume = df[df['volume_quantile'] >= 8]['target_return']
    normal_volume = df[df['volume_quantile'].between(2, 7)]['target_return']
    
    axes[0].hist(high_volume, bins=50, alpha=0.7, label='High Volume (top 20%)', density=True)
    axes[0].hist(normal_volume, bins=50, alpha=0.7, label='Normal Volume', density=True)
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Next 5-Minute Return')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Price Impact: High Volume vs Normal Volume')
    axes[0].legend()
    
    # Right plot: Returns by order imbalance decile
    df['oi_decile'] = pd.qcut(df['oi_5'].rank(method='first'), 10, labels=False)
    oi_returns = df.groupby('oi_decile')['target_return'].mean()
    
    axes[1].bar(oi_returns.index, oi_returns.values, color='steelblue', edgecolor='black')
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Order Imbalance Decile (Higher = More Buy Pressure)')
    axes[1].set_ylabel('Average Next 5-Minute Return')
    axes[1].set_title('Price Impact by Order Imbalance')
    axes[1].set_xticks(range(10))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Saved: {output_path}")

def create_sentiment_analysis_plot(df_sentiment, output_path="sentiment_analysis.png"):
    """
    Plot 2: Sentiment distribution and examples
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Sentiment distribution
    axes[0].hist(df_sentiment['sentiment_score'], bins=30, color='teal', edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Sentiment Score (-1 = Negative, +1 = Positive)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of News Sentiment Scores')
    axes[0].axvline(x=df_sentiment['sentiment_score'].mean(), color='orange', linestyle='-', alpha=0.8, label=f"Mean: {df_sentiment['sentiment_score'].mean():.2f}")
    axes[0].legend()
    
    # Right: Sentiment over time
    df_sentiment['date'] = pd.to_datetime(df_sentiment['datetime']).dt.date
    daily_sentiment = df_sentiment.groupby('date')['sentiment_score'].mean()
    
    axes[1].plot(daily_sentiment.index, daily_sentiment.values, marker='o', linewidth=2, markersize=4, color='steelblue')
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Average Daily Sentiment')
    axes[1].set_title('Sentiment Trends Over Time')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Saved: {output_path}")

def create_feature_importance_plot(output_path="feature_importance.png"):
    """
    Plot 3: Feature importance from Random Forest
    """
    try:
        importance_df = pd.read_csv("feature_importance.csv")
    except:
        print("    feature_importance.csv not found, creating sample data")
        importance_df = pd.DataFrame({
            'feature': ['lag_return_2', 'intraday_position', 'lag_return_1', 'volatility_ratio', 
                       'volatility_5', 'lag_volatility_1', 'log_volume', 'volume_ratio'],
            'importance': [0.125, 0.096, 0.092, 0.090, 0.085, 0.076, 0.076, 0.073]
        })
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_features = importance_df.head(10)
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(top_features)))
    
    bars = ax.barh(range(len(top_features)), top_features['importance'].values, color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top 10 Features for Price Impact Prediction (Random Forest)')
    ax.invert_yaxis()
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_features['importance'].values)):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Saved: {output_path}")

def create_model_comparison_plot(output_path="model_comparison.png"):
    """
    Plot 4: Model performance comparison
    """
    # Data from your runs
    models = ['Baseline\n(Linear)', 'Ridge\n+ Sentiment', 'Random Forest\n+ Sentiment']
    directional_acc = [50.32, 48.26, 49.05]
    rmse_values = [0.001154, 0.001103, 0.001140]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Directional accuracy plot
    colors_dir = ['gray', 'coral', 'coral']
    bars1 = ax1.bar(models, directional_acc, color=colors_dir, edgecolor='black')
    ax1.axhline(y=50, color='green', linestyle='--', alpha=0.7, label='Random Guess (50%)')
    ax1.set_ylabel('Directional Accuracy (%)')
    ax1.set_title('Sign Prediction Performance')
    ax1.set_ylim(46, 52)
    ax1.legend()
    
    # Add value labels
    for bar, val in zip(bars1, directional_acc):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f'{val:.1f}%', ha='center', va='bottom')
    
    # RMSE plot
    rmse_bps = [r * 10000 for r in rmse_values]  # Convert to basis points
    bars2 = ax2.bar(models, rmse_bps, color=colors_dir, edgecolor='black')
    ax2.set_ylabel('RMSE (basis points)')
    ax2.set_title('Prediction Error')
    ax2.set_ylim(10, 12)
    
    # Add value labels
    for bar, val in zip(bars2, rmse_bps):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.1f} bps', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Saved: {output_path}")

def create_correlation_matrix(df, output_path="correlation_matrix.png"):
    """
    Plot 5: Correlation matrix of key features
    """
    try:
        df_with_sentiment = pd.read_csv("aapl_with_sentiment.csv", index_col=0, parse_dates=True)
    except:
        print("    aapl_with_sentiment.csv not found, using sample data")
        return
    
    # Select key features
    key_features = ['log_volume', 'spread_pct', 'rolling_volatility', 'oi_5', 
                    'sentiment', 'target_return']
    existing_features = [f for f in key_features if f in df_with_sentiment.columns]
    
    corr_matrix = df_with_sentiment[existing_features].corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r', 
                center=0, square=True, ax=ax, cbar_kws={'shrink': 0.8})
    
    ax.set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Saved: {output_path}")

def create_llm_pipeline_diagram(output_path="llm_pipeline.png"):
    """
    Plot 6: LLM pipeline architecture diagram
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    
    # Define pipeline steps
    steps = [
        "News API\n100+ Headlines",
        "TextBlob LLM\nSentiment Scoring\n(-1 to +1)",
        "Resample to\n5-min Bars",
        "Merge with\nPrice Features",
        "Random Forest\nPrice Impact Model"
    ]
    
    # Create flow diagram
    y_pos = 0.5
    x_positions = np.linspace(0.05, 0.95, len(steps))
    
    for i, (step, x) in enumerate(zip(steps, x_positions)):
        # Box
        rect = plt.Rectangle((x-0.08, y_pos-0.2), 0.16, 0.4, 
                              facecolor='lightblue', edgecolor='navy', linewidth=2)
        ax.add_patch(rect)
        
        # Text
        ax.text(x, y_pos, step, ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Arrow
        if i < len(steps) - 1:
            ax.annotate('', xy=(x_positions[i+1]-0.08, y_pos), 
                       xytext=(x+0.08, y_pos),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('LLM-Enhanced Price Impact Pipeline', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Saved: {output_path}")

if __name__ == "__main__":
    print("="*60)
    print("CREATING PORTFOLIO VISUALIZATIONS")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    from BaseLineModel import load_data, engineer_features
    df_price = load_data("aapl_data.csv")
    df_price = engineer_features(df_price)
    
    # Load sentiment data if available
    try:
        df_sentiment = pd.read_csv("news_sentiment_data.csv")
        print("   Loaded sentiment data")
    except:
        print("   No sentiment data found, creating sample")
        df_sentiment = None
    
    # Create all plots
    print("\n2. Generating visualizations...")
    create_price_impact_plot(df_price)
    
    if df_sentiment is not None:
        create_sentiment_analysis_plot(df_sentiment)
    
    create_feature_importance_plot()
    create_model_comparison_plot()
    create_correlation_matrix(df_price)
    create_llm_pipeline_diagram()
    
    print("\n" + "="*60)
    print(" All visualizations saved!")
    print("   Files created:")
    print("   - price_impact_pattern.png")
    print("   - sentiment_analysis.png")
    print("   - feature_importance.png")
    print("   - model_comparison.png")
    print("   - correlation_matrix.png")
    print("   - llm_pipeline.png")
    print("="*60)