"""
Retrain the model using existing processed data with updated cuisine categories.
This script skips scraping and just retrains the model + recalculates residuals.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import os
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import joblib


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '../../data/processed/restaurants_with_residuals.parquet')
    
    print("Loading data...")
    gdf = gpd.read_parquet(data_path)
    print(f"Loaded {len(gdf)} restaurants")
    
    print("\nCategory distribution:")
    print(gdf['category'].value_counts())
    
    # Features for model (removed is_hawker and price_level_num)
    features = ['log_reviews', 'is_chain', 'cluster_density', 'category', 'planning_area']
    target = 'rating'
    
    # Ensure features exist
    if 'log_reviews' not in gdf.columns:
        gdf['log_reviews'] = np.log1p(gdf['review_count'].fillna(0))
    if 'is_chain' not in gdf.columns:
        name_counts = gdf['name'].value_counts()
        gdf['is_chain'] = gdf['name'].map(name_counts) > 2
    if 'cluster_density' not in gdf.columns:
        gdf['cluster_density'] = 10  # default
    
    # Filter out rows with missing values in key columns
    gdf = gdf.dropna(subset=['rating', 'category', 'planning_area'])
    print(f"\nAfter dropping NA: {len(gdf)} restaurants")
    
    X = gdf[features]
    y = gdf[target]
    
    categorical_features = ['category', 'planning_area']
    numeric_features = ['log_reviews', 'is_chain', 'cluster_density']
    
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
    
    print("\nTraining model...")
    # Cross Validation to check model stability
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    print(f"Cross-Validation RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std() * 2:.4f})")
    
    model.fit(X, y)
    
    # Feature Importance (Permutation)
    print("\nCalculating feature importance...")
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()
    
    print("\nFeature Importances:")
    for i in sorted_idx[::-1]:
        print(f"  {features[i]}: {result.importances_mean[i]:.4f}")
    
    # Calculate Residuals
    print("\nCalculating residuals...")
    gdf['predicted_rating'] = model.predict(X)
    gdf['residual'] = gdf['rating'] - gdf['predicted_rating']
    
    print("\nTop 10 Undervalued (Hidden Gems):")
    top_gems = gdf.sort_values('residual', ascending=False).head(10)
    print(top_gems[['name', 'rating', 'predicted_rating', 'residual', 'category', 'planning_area']])
    
    print("\nTop 10 Overvalued:")
    bottom = gdf.sort_values('residual', ascending=True).head(10)
    print(bottom[['name', 'rating', 'predicted_rating', 'residual', 'category', 'planning_area']])
    
    # Residual stats
    print(f"\nResidual Stats:")
    print(f"  Mean: {gdf['residual'].mean():.4f}")
    print(f"  Std: {gdf['residual'].std():.4f}")
    print(f"  Min: {gdf['residual'].min():.4f}")
    print(f"  Max: {gdf['residual'].max():.4f}")
    
    # Save
    print("\nSaving...")
    gdf.to_parquet(data_path)
    print(f"Saved results to {data_path}")
    
    # Save model
    model_path = os.path.join(script_dir, '../../data/processed/rating_model.pkl')
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")
    
    print("\n✅ Retraining complete!")


if __name__ == "__main__":
    main()
