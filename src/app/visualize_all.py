import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import os

# Paths
# Adjusted for running from project root
DATA_PATH = "sg-food-survival/data/processed/restaurants_with_residuals.parquet"
PLANNING_AREA_PATH = "sg-food-survival/data/raw/planning-areas.geojson"

def generate_map():
    print("Loading data...")
    # Load Restaurants
    df = gpd.read_parquet(DATA_PATH)
    
    # Filter for Singapore (Remove Johor)
    df = df[df.geometry.y < 1.48]
    df = df[df.geometry.y > 1.15]
    print(f"Plotting {len(df)} restaurants...")

    # Load Planning Areas for context
    if os.path.exists(PLANNING_AREA_PATH):
        planning_areas = gpd.read_file(PLANNING_AREA_PATH)
    else:
        planning_areas = None

    # Setup Plot
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Plot Planning Areas (Base)
    if planning_areas is not None:
        planning_areas.to_crs(epsg=3857).plot(ax=ax, facecolor='none', edgecolor='#333', linewidth=0.5, alpha=0.5)
        
    # Plot Restaurants
    # Convert to Web Mercator for Contextily
    df_wm = df.to_crs(epsg=3857)
    
    # Plot points
    # Color by residual (Red=Overvalued, Green=Undervalued) just for fun, or just simple dots
    # User just asked for "depicted as dots", so simple black or blue dots is fine.
    # Let's use a small marker size.
    df_wm.plot(ax=ax, markersize=1, color='#2a9d8f', alpha=0.5, label='Restaurants')

    # Add Basemap
    try:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    except:
        print("Could not fetch basemap, skipping.")

    ax.set_axis_off()
    plt.title(f"All {len(df)} Scraped Restaurants in Singapore", fontsize=20)
    
    output_path = "all_restaurants_map.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Map saved to {output_path}")

if __name__ == "__main__":
    generate_map()
