# Price Impact Model with LLM Sentiment Integration

## Project Overview
Quantitative research project built for internship opportunities. Predicts short-term price impact using market microstructure features and LLM-based sentiment analysis.

## Key Findings
- Public news sentiment shows near-zero correlation with 5-minute returns (ρ = -0.0075)
- Adding sentiment does not improve directional accuracy (49.05% vs 50.32% baseline)
- Confirms efficient market hypothesis for public data

## Technical Stack
- Python (pandas, numpy, scikit-learn)
- LLM: TextBlob sentiment analysis
- Visualization: matplotlib, seaborn
- Data: Yahoo Finance (AAPL, 1,583 rows, 5-min bars)


## PROJECT WORKFLOW

This document outlines the step-by-step workflow for the entire project.
Follow these steps in order to reproduce all results.

-------------------------------------------------------------------------------
                  Step-1: Data Acquisition
-------------------------------------------------------------------------------

File: Get_data.py

What it does:
- Downloads AAPL 5-minute OHLCV data from Yahoo Finance
- Period: 1 month (May 4, 2026 - June 2, 2026)
- Interval: 5-minute bars

Output files created:
- aapl_data.csv (1,583 rows of raw price/volume data)

-------------------------------------------------------------------------------
                  Step-2: Baseline Linear Model
-------------------------------------------------------------------------------

File: BaseLineModel.py

What it does:
- Loads data from aapl_data.csv
- Engineers basic features:
    * Simulated bid/ask from High/Low
    * Order imbalance (5-period rolling = 25 minutes)
    * Rolling volatility, spread, log volume
- Trains linear regression model
- Evaluates performance (RMSE, R², directional accuracy)

Output files created:
- baseline_results.csv (model performance metrics)

How to run:
    python BaseLineModel.py

-------------------------------------------------------------------------------
                  Step-3: Enhanced Non-linear Models
-------------------------------------------------------------------------------

File: EnhancedModel.py

What it does:
- Loads data from aapl_data.csv
- Engineers advanced features:
    * Multiple OI windows (15, 25, 50, 75 minutes)
    * Lagged features (returns, volume, OI, volatility)
    * Interaction features (volume×spread, volatility×OI)
    * Time features (intraday position)
- Trains multiple models:
    * Ridge Regression (L2 regularization)
    * Random Forest
    * Gradient Boosting
- Uses Time Series Cross-Validation (no look-ahead)

Output files created:
- enhanced_results.csv (model comparison)
- feature_importance.csv (top predictors)


Key insight:
    Ridge shows slight RMSE improvement (4.2%) but directional accuracy remains near 50%.

-------------------------------------------------------------------------------
                  Step-4: LLM Sentiment Integration (No Look-ahead Bias)
-------------------------------------------------------------------------------

File: LLM_Sentiment.py

What it does:
- Fetches 100+ news headlines from NewsAPI (requires free API key)
- Computes sentiment scores using TextBlob LLM
- ALIGNS SENTIMENT WITH FUTURE RETURNS (critical - avoids look-ahead bias)
    * News at 14:32 predicts 14:35-14:40 bar, NOT 14:30-14:35 bar
- Adds sentiment features to model
- Trains Ridge and Random Forest with sentiment

CRITICAL METHODOLOGICAL NOTE:
    This implementation ensures NO LOOK-AHEAD BIAS by only using news
    published BEFORE each 5-minute bar started.

Output files created:
- aapl_with_sentiment_no_bias.csv (price + sentiment features)
- news_sentiment_data.csv (headlines with sentiment scores)

How to run:
    # First, get a free API key from https://newsapi.org/register
    # Then replace YOUR_API_KEY_HERE in the file
    python LLM_Sentiment.py

Key finding:
    Public news sentiment provides NO short-term alpha - consistent with
    efficient market hypothesis.

-------------------------------------------------------------------------------
                  Step-5: Visualization
-------------------------------------------------------------------------------

File: Visualization.py

What it does:
- Creates 6 professional plots for portfolio/README
- Visualizes price impact patterns, sentiment analysis, feature importance

Output files created:
- price_impact_pattern.png (returns after high volume vs normal volume)
- sentiment_analysis.png (sentiment distribution and trends)
- feature_importance.png (top 10 predictors from Random Forest)
- model_comparison.png (bar chart comparing all models)
- correlation_matrix.png (feature correlations heatmap)
- llm_pipeline.png (LLM architecture diagram)


-------------------------------------------------------------------------------
                 File Structure
-------------------------------------------------------------------------------






Project
- Code Files:
  - Get_data.py                 # Step 1: Data download
  - BaseLineModel.py            # Step 2: Baseline linear model
  - EnhancedModel.py            # Step 3: Non-linear models
  - LLM_Sentiment.py            # Step 4: LLM integration
  - Visualization.py            # Step 5: Generate plots
  Data Files:
    aapl_data.csv               # Raw price data (1,583 rows)
    aapl_with_sentiment_no_bias.csv  # Enhanced with sentiment
    news_sentiment_data.csv     # Headlines with scores
  Results Files:
    baseline_results.csv        # Linear model metrics
    enhanced_results.csv        # Non-linear model comparison
    feature_importance.csv      # Top predictors
  Visualizations (6 PNG files):
    price_impact_pattern.png
    sentiment_analysis.png
    feature_importance.png
    model_comparison.png
    correlation_matrix.png
    llm_pipeline.png
