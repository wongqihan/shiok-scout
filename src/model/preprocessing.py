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

        # --- Manual Coordinate Fixes ---
        # Some restaurants have incorrect scraped coordinates (e.g. Luss in Changi instead of Queenstown)
        
        # Luss Restaurant & Bar (Portsdown Rd -> Queenstown)
        mask_luss = gdf['name'].str.contains('Luss Restaurant', case=False, na=False)
        if mask_luss.any():
            gdf.loc[mask_luss, 'latitude'] = 1.2915
            gdf.loc[mask_luss, 'longitude'] = 103.7925
            # Update geometry
            gdf.loc[mask_luss, 'geometry'] = gpd.points_from_xy(gdf.loc[mask_luss, 'longitude'], gdf.loc[mask_luss, 'latitude'])

        # 9 Plus Bistro (Pasir Panjang -> Queenstown/Bukit Merah)
        mask_9plus = gdf['name'].str.contains('9 Plus Bistro', case=False, na=False)
        if mask_9plus.any():
            gdf.loc[mask_9plus, 'latitude'] = 1.2830
            gdf.loc[mask_9plus, 'longitude'] = 103.7810
            gdf.loc[mask_9plus, 'geometry'] = gpd.points_from_xy(gdf.loc[mask_9plus, 'longitude'], gdf.loc[mask_9plus, 'latitude'])

        # Wu You Eating Place (Tuas -> Lavender/Kallang)
        mask_wuyou = gdf['name'].str.contains('Wu You Eating Place', case=False, na=False)
        if mask_wuyou.any():
            gdf.loc[mask_wuyou, 'latitude'] = 1.3130
            gdf.loc[mask_wuyou, 'longitude'] = 103.8610
            gdf.loc[mask_wuyou, 'geometry'] = gpd.points_from_xy(gdf.loc[mask_wuyou, 'longitude'], gdf.loc[mask_wuyou, 'latitude'])

        # Oasis Hideout (Lim Chu Kang -> Dover/Queenstown)
        mask_oasis = gdf['name'].str.contains('Oasis Hideout', case=False, na=False)
        if mask_oasis.any():
            gdf.loc[mask_oasis, 'latitude'] = 1.3050
            gdf.loc[mask_oasis, 'longitude'] = 103.7800
            gdf.loc[mask_oasis, 'geometry'] = gpd.points_from_xy(gdf.loc[mask_oasis, 'longitude'], gdf.loc[mask_oasis, 'latitude'])

        # L Bistro (Western Islands -> Jurong East)
        mask_lbistro = gdf['name'].str.contains('L Bistro', case=False, na=False)
        if mask_lbistro.any():
            gdf.loc[mask_lbistro, 'latitude'] = 1.3330
            gdf.loc[mask_lbistro, 'longitude'] = 103.7430
            gdf.loc[mask_lbistro, 'geometry'] = gpd.points_from_xy(gdf.loc[mask_lbistro, 'longitude'], gdf.loc[mask_lbistro, 'latitude'])

        # --- Spatial Join ---
        # Ensure CRS matches (EPSG:4326 for lat/lon, but we project for distance)
        # Planning areas are usually in 3414 or 4326. Let's check.
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
            
            # Split unknowns into Mainland (Lat <= 1.44) and Border (Lat > 1.44)
            # Mainland can have a large buffer (2km) to catch points in gaps/roads.
            # Border needs a tight buffer (100m) to exclude JB.
            
            # Ensure we have latitude. If not, extract from geometry.
            if 'latitude' not in gdf_unknown.columns:
                gdf_unknown['latitude'] = gdf_unknown.geometry.y
                
            mask_border = gdf_unknown['latitude'] > 1.44
            gdf_border = gdf_unknown[mask_border].copy()
            gdf_mainland = gdf_unknown[~mask_border].copy()
            
            # Project
            gdf_border_proj = gdf_border.to_crs("EPSG:3414")
            gdf_mainland_proj = gdf_mainland.to_crs("EPSG:3414")
            planning_area_proj = planning_area_gdf.to_crs("EPSG:3414")
            
            # Pass 1: Border (Strict 100m)
            if not gdf_border.empty:
                print(f"Matching {len(gdf_border)} border points with 100m buffer...")
                gdf_nearest_border = gpd.sjoin_nearest(
                    gdf_border_proj, 
                    planning_area_proj[['name', 'geometry']], 
                    how='left', 
                    max_distance=100, 
                    distance_col='dist_to_area'
                )
                # Deduplicate
                gdf_nearest_border = gdf_nearest_border[~gdf_nearest_border.index.duplicated(keep='first')]
                
                # Update original
                # We need to handle the column renaming if sjoin_nearest added suffix
                col_name = 'name_right' if 'name_right' in gdf_nearest_border.columns else 'planning_area'
                if col_name in gdf_nearest_border.columns:
                     gdf_joined.loc[gdf_nearest_border.index, 'planning_area'] = gdf_nearest_border[col_name]

            # Pass 2: Mainland (Generous 2000m)
            if not gdf_mainland.empty:
                print(f"Matching {len(gdf_mainland)} mainland points with 2000m buffer...")
                gdf_nearest_mainland = gpd.sjoin_nearest(
                    gdf_mainland_proj, 
                    planning_area_proj[['name', 'geometry']], 
                    how='left', 
                    max_distance=2000, 
                    distance_col='dist_to_area'
                )
                # Deduplicate
                gdf_nearest_mainland = gdf_nearest_mainland[~gdf_nearest_mainland.index.duplicated(keep='first')]
                
                # Update original
                col_name = 'name_right' if 'name_right' in gdf_nearest_mainland.columns else 'planning_area'
                if col_name in gdf_nearest_mainland.columns:
                     gdf_joined.loc[gdf_nearest_mainland.index, 'planning_area'] = gdf_nearest_mainland[col_name]
            
        # Fill remaining NaNs (too far away) with 'Outside Singapore'
        gdf_joined['planning_area'] = gdf_joined['planning_area'].fillna('Outside Singapore')
        
        # HARD CUTOFF for JB
        # Any point with latitude > 1.46 is likely Johor Bahru (Woodlands max is ~1.462 but mostly < 1.46)
        # Actually, let's just rely on the 100m buffer.
        # But keep a safety net for deep JB.
        if 'latitude' in gdf_joined.columns:
            gdf_joined.loc[gdf_joined['latitude'] > 1.47, 'planning_area'] = 'Outside Singapore'
        elif hasattr(gdf_joined.geometry, 'y'):
             gdf_joined.loc[gdf_joined.geometry.y > 1.47, 'planning_area'] = 'Outside Singapore'

        gdf = gdf_joined
    else:
        gdf['planning_area'] = 'Unknown'

    print("Preprocessing complete.")
    return gdf
