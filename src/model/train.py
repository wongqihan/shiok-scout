import pandas as pd
import geopandas as gpd
import numpy as np
import os
import sys
import time
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import joblib

# Add src to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.scraper.google_maps import RestaurantScraper
from src.model.preprocessing import preprocess_data

RAW_DATA_DIR = "../../data/raw"
PROCESSED_DATA_DIR = "../../data/processed"

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 1. Scrape Data
    # NOTE: Set use_dummy=False to use the real Google Maps scraper (requires Playwright)
    # User requested to try Playwright again.
    USE_DUMMY = False 
    
    if USE_DUMMY:
        print("Generating DUMMY data (Set USE_DUMMY=False to scrape real data)...")
        scraper = RestaurantScraper(use_dummy=True)
        # Generate more data for a better model simulation
        seeds_path = os.path.join(script_dir, PROCESSED_DATA_DIR, "search_seeds.geojson")
        if os.path.exists(seeds_path):
            seeds = gpd.read_file(seeds_path)
            sample_seeds = seeds.sample(n=50, random_state=42)
        else:
            sample_seeds = pd.DataFrame({'geometry': [], 'latitude': [1.3521], 'longitude': [103.8198]})
    else:
        print("Scraping REAL data from Google Maps...")
        scraper = RestaurantScraper(use_dummy=False, headless=True)
        seeds_path = os.path.join(script_dir, PROCESSED_DATA_DIR, "search_seeds.geojson")
        if os.path.exists(seeds_path):
            seeds = gpd.read_file(seeds_path)
            # Use ALL seeds for complete scrape
            sample_seeds = seeds
            print(f"Selected ALL {len(sample_seeds)} seeds for scraping. This will take a while.")
        else:
            sample_seeds = pd.DataFrame({'geometry': [], 'latitude': [1.3521], 'longitude': [103.8198]})

    all_restaurants = []
    raw_output_path = os.path.join(script_dir, PROCESSED_DATA_DIR, "restaurants_raw.parquet")
    
    # Resume Logic
    processed_indices = set()
    if os.path.exists(raw_output_path):
        try:
            existing_df = pd.read_parquet(raw_output_path)
            all_restaurants = existing_df.to_dict('records')
            print(f"Resuming... Loaded {len(all_restaurants)} existing items.")
            # We need to know which seeds were processed.
            # Since we didn't save seed ID, we can approximate by checking if we have data near the seed?
            # Or simpler: we just assume the loop order is deterministic (it is, if we don't shuffle or use same seed).
            # But we are iterating by index.
            # Let's just save a separate "processed_seeds.txt" or similar?
            # Or better, just rely on the fact that we save every 50.
            # If we have N items, and average M items per seed... it's hard to map back exactly.
            
            # Alternative: Just restart from seed index = (number of items / avg items per seed)? No, too risky.
            # Let's just skip seeds that are close to existing points?
            # Too complex.
            
            # Let's implement a 'processed_seeds.json' log.
            processed_seeds_path = os.path.join(script_dir, PROCESSED_DATA_DIR, "processed_seeds.json")
            if os.path.exists(processed_seeds_path):
                import json
                with open(processed_seeds_path, 'r') as f:
                    processed_indices = set(json.load(f))
                print(f"Found {len(processed_indices)} processed seeds in log.")
        except Exception as e:
            print(f"Error loading existing data: {e}. Starting fresh.")
            
    total_seeds = len(sample_seeds)
    import json
    processed_seeds_path = os.path.join(script_dir, PROCESSED_DATA_DIR, "processed_seeds.json")

    for idx, row in sample_seeds.iterrows():
        if idx in processed_indices:
            continue

        # Handle Point geometry
        if hasattr(row.geometry, 'y'):
            lat, lon = row.geometry.y, row.geometry.x
        else:
            lat, lon = row['latitude'], row['longitude']
            
        print(f"Scraping seed {idx+1}/{total_seeds}: {lat:.5f}, {lon:.5f}...")
        
        # Restart scraper every 100 iterations to prevent OOM
        if (len(processed_indices) + 1) % 100 == 0:
            print("Restarting scraper to free memory...")
            scraper.close()
            time.sleep(2)
            scraper = RestaurantScraper(use_dummy=False, headless=True)

        try:
            results = scraper.scrape_location(lat, lon)
            all_restaurants.extend(results)
        except Exception as e:
            print(f"Failed to scrape seed {idx}: {e}")
        
        processed_indices.add(idx)

        # Intermediate save every 50 seeds
        if len(processed_indices) % 50 == 0:
            print(f"Saving partial results ({len(all_restaurants)} items)...")
            pd.DataFrame(all_restaurants).to_parquet(raw_output_path)
            with open(processed_seeds_path, 'w') as f:
                json.dump(list(processed_indices), f)
        
    scraper.close()
    
    if not all_restaurants:
        print("No restaurants found.")
        return
        
    df = pd.DataFrame(all_restaurants)
    # Save final raw data
    df.to_parquet(raw_output_path)
    print(f"Collected {len(df)} restaurants. Saved raw data to {raw_output_path}")
    
    # 2. Load Hawker Centres
    hawker_path = os.path.join(script_dir, RAW_DATA_DIR, "hawker-centres.geojson")
    if not os.path.exists(hawker_path):
         hawker_path = os.path.join(script_dir, RAW_DATA_DIR, "hawker-centres.kml")
    
    try:
        hawker_gdf = gpd.read_file(hawker_path)
    except Exception as e:
        print(f"Error loading hawker centres: {e}")
        return

    # Load Planning Areas
    planning_area_path = os.path.join(script_dir, RAW_DATA_DIR, "planning-areas.geojson")
    if os.path.exists(planning_area_path):
        planning_area_gdf = gpd.read_file(planning_area_path)
    else:
        planning_area_gdf = None

    # 3. Preprocess
    gdf = preprocess_data(df, hawker_gdf, planning_area_gdf)
    
    # 4. Train Model with Refinements
    features = ['log_reviews', 'price_level_num', 'is_chain', 'cluster_density', 'category_encoded', 'planning_area']
    target = 'rating'
    
    X = gdf[features]
    y = gdf[target]
    
    categorical_features = ['category_encoded', 'planning_area']
    numeric_features = ['log_reviews', 'price_level_num', 'is_chain', 'cluster_density']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features),
            ('num', 'passthrough', numeric_features)
        ]
    )
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', HistGradientBoostingRegressor(categorical_features=[0, 1], l2_regularization=1.0))
    ])
    
    print("Training model...")
    # Cross Validation to check model stability
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    print(f"Cross-Validation RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std() * 2:.4f})")
    
    model.fit(X, y)
    
    # Feature Importance (Permutation)
    print("Calculating feature importance...")
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=1)
    sorted_idx = result.importances_mean.argsort()
    
    print("\nFeature Importances:")
    for i in sorted_idx[::-1]:
        print(f"{features[i]}: {result.importances_mean[i]:.4f}")
    
    # 5. Calculate Residuals
    gdf['predicted_rating'] = model.predict(X)
    gdf['residual'] = gdf['rating'] - gdf['predicted_rating']
    
    print("Top 5 Undervalued (Hidden Gems):")
    print(gdf.sort_values('residual', ascending=False).head(5)[['name', 'rating', 'predicted_rating', 'residual', 'planning_area']])
    
    # 6. Save
    output_path = os.path.join(script_dir, PROCESSED_DATA_DIR, "restaurants_with_residuals.parquet")
    # Convert to standard dataframe for parquet if geometry causes issues, but geopandas supports parquet
    # However, streamlit might prefer standard pandas or geojson.
    # Let's save as parquet (geoparquet)
    gdf.to_parquet(output_path)
    print(f"Saved results to {output_path}")
    
    # Save model
    model_path = os.path.join(script_dir, PROCESSED_DATA_DIR, "rating_model.pkl")
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")

if __name__ == "__main__":
    main()
