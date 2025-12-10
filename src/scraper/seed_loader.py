import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import os

# Define paths relative to this script
RAW_DATA_DIR = "../../data/raw"
PROCESSED_DATA_DIR = "../../data/processed"

def load_hawker_centres(kml_path):
    """Loads Hawker Centres from KML and returns a GeoDataFrame."""
    # Note: KML support in fiona/geopandas can be tricky. 
    # If this fails, we might need to enable KML driver or use fastkml.
    # For now, we attempt standard read.
    try:
        # Try reading with default driver
        gdf = gpd.read_file(kml_path)
    except Exception as e:
        print(f"Failed to read KML with default driver: {e}")
        print("Attempting to read GeoJSON if available as fallback.")
        # Fallback to GeoJSON if KML fails
        geojson_path = kml_path.replace(".kml", ".geojson")
        if os.path.exists(geojson_path):
            print(f"Found GeoJSON at {geojson_path}, loading that instead.")
            gdf = gpd.read_file(geojson_path)
        else:
            # If no GeoJSON, try specifying KML driver explicitly if fiona supports it
            try:
                gdf = gpd.read_file(kml_path, driver='KML')
            except Exception as e2:
                print(f"Failed to read with KML driver: {e2}")
                raise e
    return gdf

def generate_grid(spacing_meters=1000):
    """Generates a grid of points over Singapore."""
    # Singapore Bounding Box (approx)
    # 1.15, 103.6 to 1.47, 104.05
    
    lat_min, lat_max = 1.20, 1.48 
    lon_min, lon_max = 103.60, 104.05
    
    # 1 degree lat approx 111km. 
    # 1000m = 1km = 1/111 degrees approx 0.009 degrees.
    step = 0.009 * (spacing_meters / 1000.0)
    
    lats = np.arange(lat_min, lat_max, step)
    lons = np.arange(lon_min, lon_max, step)
    
    grid_points = []
    for lat in lats:
        for lon in lons:
            grid_points.append(Point(lon, lat))
            
    return gpd.GeoDataFrame(geometry=grid_points, crs="EPSG:4326")

def main():
    # 1. Load Hawker Centres
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kml_path = os.path.join(script_dir, RAW_DATA_DIR, "hawker-centres.kml")
    
    print(f"Loading Hawker Centres from {kml_path}...")
    if not os.path.exists(kml_path):
        print(f"Error: File not found at {kml_path}")
        return

    hawker_gdf = load_hawker_centres(kml_path)
    
    # Extract centroids
    # KML often contains 3D coordinates (x, y, z). We need to ensure we flatten or handle them.
    # Also, hawker centres might be Polygons or Points.
    # Warning: centroid of a Point is the Point itself, so this is safe.
    hawker_gdf = hawker_gdf.to_crs("EPSG:4326") # Ensure WGS84
    hawker_gdf['centroid'] = hawker_gdf.geometry.centroid
    hawker_points = hawker_gdf[['centroid']].copy()
    hawker_points = hawker_points.rename(columns={'centroid': 'geometry'}).set_geometry('geometry')
    hawker_points['type'] = 'hawker_centre'
    
    print(f"Loaded {len(hawker_points)} hawker centres.")
    
    # 2. Generate Grid
    # User requested approx 500m radius coverage. 
    # We'll use 800m spacing to ensure good coverage without excessive overlap if radius is 500m.
    # Or just 500m spacing for dense coverage. Let's go with 700m.
    print("Generating search grid...")
    grid_gdf = generate_grid(spacing_meters=700) 
    grid_gdf['type'] = 'grid_point'
    print(f"Generated {len(grid_gdf)} grid points.")
    
    # 3. Combine
    all_seeds = pd.concat([hawker_points, grid_gdf], ignore_index=True)
    
    # 4. Save
    output_path = os.path.join(script_dir, PROCESSED_DATA_DIR, "search_seeds.geojson")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    all_seeds.to_file(output_path, driver='GeoJSON')
    print(f"Saved {len(all_seeds)} seeds to {output_path}")

if __name__ == "__main__":
    main()
