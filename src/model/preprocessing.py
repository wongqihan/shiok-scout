import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.neighbors import BallTree

def preprocess_data(df: pd.DataFrame, hawker_gdf: gpd.GeoDataFrame, planning_area_gdf: gpd.GeoDataFrame = None) -> pd.DataFrame:
    """
    Transforms raw restaurant data into features for the model.
    """
    print("Preprocessing data...")
    
    # Ensure df is a GeoDataFrame
    if not isinstance(df, gpd.GeoDataFrame):
        # Check if we have lat/lon
        if 'latitude' in df.columns and 'longitude' in df.columns:
            gdf = gpd.GeoDataFrame(
                df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326"
            )
        else:
            raise ValueError("Dataframe must have latitude and longitude columns")
    else:
        gdf = df.copy()

    # 1. Log Reviews
    # Add 1 to avoid log(0)
    gdf['log_reviews'] = np.log1p(gdf['review_count'].fillna(0))

    # 2. Is_Chain
    # Heuristic: if name appears > 2 times in dataset
    name_counts = gdf['name'].value_counts()
    gdf['is_chain'] = gdf['name'].map(name_counts) > 2

    # 3. Is_Hawker
    # Check distance to nearest hawker centre centroid
    # Project to SVY21 (EPSG:3414) for meters
    if hawker_gdf.crs != "EPSG:3414":
        hawker_proj = hawker_gdf.to_crs("EPSG:3414")
    else:
        hawker_proj = hawker_gdf
        
    if gdf.crs != "EPSG:3414":
        gdf_proj = gdf.to_crs("EPSG:3414")
    else:
        gdf_proj = gdf
    
    # Use BallTree for efficient nearest neighbor search
    # Extract coordinates (x, y)
    hawker_coords = np.column_stack((hawker_proj.geometry.x, hawker_proj.geometry.y))
    restaurant_coords = np.column_stack((gdf_proj.geometry.x, gdf_proj.geometry.y))
    
    if len(hawker_coords) > 0:
        tree = BallTree(hawker_coords, metric='euclidean')
        dist, _ = tree.query(restaurant_coords, k=1) # dist is in meters
        gdf['dist_to_hawker'] = dist.flatten()
        gdf['is_hawker'] = gdf['dist_to_hawker'] < 50.0 # 50m threshold
    else:
        gdf['is_hawker'] = False
        gdf['dist_to_hawker'] = 9999.0

    # 4. Category Encoded
    # Keep top 20 cuisines, others as 'Other'
    if 'category' in gdf.columns:
        top_categories = gdf['category'].value_counts().nlargest(20).index
        gdf['category_encoded'] = gdf['category'].apply(lambda x: x if x in top_categories else 'Other')
    else:
        gdf['category_encoded'] = 'Other'

    # 5. Cluster Density
    # Calculate how many other food spots are within a 200m radius
    tree_self = BallTree(restaurant_coords, metric='euclidean')
    counts = tree_self.query_radius(restaurant_coords, r=200.0, count_only=True)
    gdf['cluster_density'] = counts - 1 # Exclude self

    # 6. Price Level
    # Convert $ to 1, $$ to 2, etc.
    price_map = {'$': 1, '$$': 2, '$$$': 3, '$$$$': 4}
    if 'price_level' in gdf.columns:
        gdf['price_level_num'] = gdf['price_level'].map(price_map).fillna(1)
    else:
        gdf['price_level_num'] = 1

    # 7. Planning Area
    if planning_area_gdf is not None:
        # Ensure CRS matches
        if planning_area_gdf.crs != gdf.crs:
            planning_area_gdf = planning_area_gdf.to_crs(gdf.crs)
            
        # Spatial Join
        # 'inner' or 'left'. Left ensures we keep all restaurants even if they fall outside (e.g. sea)
        gdf_joined = gpd.sjoin(gdf, planning_area_gdf[['name', 'geometry']], how='left', predicate='within')
        
        # Rename 'name_right' (from planning area) to 'planning_area'
        # Note: sjoin usually produces index_right and columns from right df.
        # If 'name' is in both, it becomes name_left and name_right.
        if 'name_right' in gdf_joined.columns:
            gdf_joined = gdf_joined.rename(columns={'name_right': 'planning_area'})
            # Drop name_left and rename back to name if needed, or just keep name_left as name
            if 'name_left' in gdf_joined.columns:
                gdf_joined = gdf_joined.rename(columns={'name_left': 'name'})
        elif 'name' in planning_area_gdf.columns and 'name' not in gdf.columns:
             gdf_joined = gdf_joined.rename(columns={'name': 'planning_area'})
        
        # Fill NaNs
        gdf_joined['planning_area'] = gdf_joined['planning_area'].fillna('Unknown')
        
        # Clean up sjoin artifacts
        cols_to_drop = ['index_right']
        gdf = gdf_joined.drop(columns=[c for c in cols_to_drop if c in gdf_joined.columns])
    else:
        gdf['planning_area'] = 'Unknown'

    print("Preprocessing complete.")
    return gdf
