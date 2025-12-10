# Shiok Scout - Singapore Hidden Gem Restaurant Finder

🍜 **Find algorithmically undervalued restaurants in Singapore**

An interactive map that uses machine learning to find restaurants that outperform expectations for their context - true "hidden gems" that traditional rating lists miss.

## Features

- **Residual-based scoring**: Compares actual ratings vs predicted ratings based on location, cuisine, popularity, and competition
- **Interactive map**: Color-coded dots (teal = hidden gem, yellow = fair value, coral = overvalued)
- **Smart tooltips**: Each restaurant includes an AI-generated explanation of why it's rated that way
- **Filters**: Filter by planning area, cuisine type, minimum rating, and "gems only"

## How It Works

1. **Data Collection**: Scraped 70,000+ restaurant entries from Google Maps
2. **Cuisine Classification**: Used Gemini API to classify restaurants by cuisine type
3. **ML Model**: HistGradientBoostingRegressor predicts expected rating based on features
4. **Residual Calculation**: `residual = actual_rating - predicted_rating`

## Tech Stack

- Python, Streamlit, PyDeck
- scikit-learn (HistGradientBoostingRegressor)
- Google Gemini API (cuisine classification)
- Playwright (web scraping)

## Run Locally

```bash
pip install -r requirements.txt
streamlit run src/app/main.py
```

## Live Demo

[Try Shiok Scout](https://shiok-scout.streamlit.app)
