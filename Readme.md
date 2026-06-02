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

## Repository Structure
