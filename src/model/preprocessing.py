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
        # Drop existing planning_area if present to avoid duplicates after join
        if 'planning_area' in gdf.columns:
            gdf = gdf.drop(columns=['planning_area'])
            
        # Ensure CRS matches
        if planning_area_gdf.crs != gdf.crs:
            planning_area_gdf = planning_area_gdf.to_crs(gdf.crs)
            
        # Step 1: Exact spatial join (point within polygon)
        gdf_joined = gpd.sjoin(gdf, planning_area_gdf[['name', 'geometry']], how='left', predicate='within')
        
        # Rename 'name_right' to 'planning_area'
        if 'name_right' in gdf_joined.columns:
            gdf_joined = gdf_joined.rename(columns={'name_right': 'planning_area'})
        elif 'name' in planning_area_gdf.columns:
             # If sjoin didn't prefix because name wasn't in left, but we want to be safe
             # Actually sjoin usually adds _right if collision.
             # If no collision, it keeps 'name'. But 'name' is in restaurants (restaurant name).
             # So it should be 'name_right'.
             pass
             
        # Handle cases where 'name' collision might have happened differently or not at all
        # If 'planning_area' column doesn't exist yet, look for 'name_right' or try to find where the area name went.
        # In the previous code, we had logic for this. Let's be robust.
        # The planning_area_gdf has 'name'. The restaurant gdf has 'name'.
        # So sjoin will produce 'name_left' and 'name_right'.
        if 'planning_area' not in gdf_joined.columns:
             if 'name_right' in gdf_joined.columns:
                 gdf_joined = gdf_joined.rename(columns={'name_right': 'planning_area'})
        
        # Restore 'name' from 'name_left' if it got renamed
        if 'name_left' in gdf_joined.columns:
            gdf_joined = gdf_joined.rename(columns={'name_left': 'name'})
            
        # Clean up sjoin artifacts
        cols_to_drop = ['index_right']
        gdf_joined = gdf_joined.drop(columns=[c for c in cols_to_drop if c in gdf_joined.columns])
        
        # Step 2: Handle Unknowns with Nearest Neighbor
        # Identify rows where planning_area is NaN
        unknown_mask = gdf_joined['planning_area'].isna()
        
        if unknown_mask.any():
            print(f"Found {unknown_mask.sum()} locations outside exact planning areas. Attempting nearest match...")
            
            # Separate unknown rows
            gdf_unknown = gdf_joined[unknown_mask].copy()
            # Drop the NaN planning_area column to avoid conflict in next join
            gdf_unknown = gdf_unknown.drop(columns=['planning_area'])
            
            # Project to meters for distance calculation (EPSG:3414)
            # We need to project both for accurate distance
            gdf_unknown_proj = gdf_unknown.to_crs("EPSG:3414")
            planning_area_proj = planning_area_gdf.to_crs("EPSG:3414")
            
            # Nearest join with max distance (e.g., 500m to catch coastal spots, but exclude JB)
            # 500 meters buffer
            gdf_nearest = gpd.sjoin_nearest(
                gdf_unknown_proj, 
                planning_area_proj[['name', 'geometry']], 
                how='left', 
                max_distance=500, 
                distance_col='dist_to_area'
            )
            
            # Rename columns back
            if 'name_right' in gdf_nearest.columns:
                gdf_nearest = gdf_nearest.rename(columns={'name_right': 'planning_area'})
            
            # Filter out points that are too far (still NaN or > threshold)
            # sjoin_nearest with max_distance leaves them out or includes them? 
            # If how='left', it keeps them but columns are NaN if no match within distance.
            
            # Update the original joined dataframe
            # We need to map the results back.
            # Create a mapping dictionary: index -> planning_area
            # Note: sjoin_nearest might return multiple matches if equidistant? 
            # We take the first one per index.
            gdf_nearest = gdf_nearest[~gdf_nearest.index.duplicated(keep='first')]
            
            # Update values in gdf_joined
            # We only update where it was unknown
            # Using combine_first or direct assignment
            gdf_joined.loc[unknown_mask, 'planning_area'] = gdf_nearest['planning_area']
            
        # Fill remaining NaNs (too far away) with 'Outside Singapore'
        gdf_joined['planning_area'] = gdf_joined['planning_area'].fillna('Outside Singapore')
        
        # HARD CUTOFF for JB
        # Any point with latitude > 1.455 is likely Johor Bahru, even if it matched 'Woodlands' via nearest neighbor
        # (Woodlands Checkpoint is ~1.445)
        if 'latitude' in gdf_joined.columns:
            gdf_joined.loc[gdf_joined['latitude'] > 1.455, 'planning_area'] = 'Outside Singapore'
        elif hasattr(gdf_joined.geometry, 'y'):
             # If geometry is available (it should be)
             gdf_joined.loc[gdf_joined.geometry.y > 1.455, 'planning_area'] = 'Outside Singapore'

        gdf = gdf_joined
    else:
        gdf['planning_area'] = 'Unknown'

    print("Preprocessing complete.")
    return gdf
