import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import numpy as np
import os

# Page Config
st.set_page_config(
    page_title="🍜 Shiok Scout",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar

# Constants
DATA_PATH = "../../data/processed/restaurants_with_residuals.parquet"

# Custom CSS for Floating Panel & Fullscreen Map
st.markdown("""
<style>
    /* 1. Reset Main Container to Fullscreen */
    .block-container {
        padding: 0rem !important;
        max-width: 100% !important;
    }
    
    [data-testid="stAppViewContainer"] {
        padding: 0 !important;
        margin: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        overflow: hidden !important;
    }
    
    /* 2. Hide Header & Footer */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stHeader"] {display: none;}
    
    /* 3. Make Sidebar Transparent and Overlay Map */
    section[data-testid="stSidebar"] {
        position: absolute !important;
        top: 0;
        left: 0;
        width: 100% !important;
        height: 100vh !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        z-index: 1000 !important;
        pointer-events: none;
        padding-top: 0 !important;
    }
    
    /* 4. Style the Inner Content of the Sidebar as the Floating Card */
    section[data-testid="stSidebar"] > div {
        width: 350px !important;
        background-color: white !important;
        margin-top: 10px;
        margin-left: 20px;
        margin-bottom: 0;
        padding: 18px;
        padding-top: 12px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        pointer-events: auto;
        height: auto !important;
        max-height: 92vh;
        overflow-y: auto;
        border: 1px solid #e0e0e0;
    }
    
    /* 5. Force Main Content to Full Width/Height (Behind Sidebar) */
    section[data-testid="stMain"] {
        width: 100vw !important;
        height: 100vh !important;
        position: fixed !important;
        top: 0;
        left: 0;
        z-index: 1;
        padding: 0 !important;
    }
    
    /* 6. Fix Text Colors in Sidebar (Force Dark Text) */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: #333333 !important;
    }
    
    /* Fix Input Fields Contrast */
    [data-testid="stSidebar"] input {
        color: #333333 !important;
        background-color: #f8f9fa !important;
        border: 1px solid #ced4da !important;
    }
    
    /* Fix Multiselect & Selectbox */
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #f8f9fa !important;
        color: #333333 !important;
        border-color: #ced4da !important;
    }
    
    /* Fix Dropdown Options */
    [data-baseweb="menu"] {
        background-color: white !important;
    }
    [data-baseweb="menu"] div {
        color: #333333 !important;
    }

    /* 7. Ensure Map takes full space */
    iframe {
        width: 100vw !important;
        height: 100vh !important;
        display: block;
    }
    
    /* Hide the button to collapse sidebar */
    [data-testid="stSidebarCollapseButton"] {
        display: none;
    }

    /* 8. Fix Placeholder Color */
    input::placeholder {
        color: #666 !important;
        opacity: 1 !important;
    }
    
    /* 9. Fix Toggle Switch Visibility */
    /* Ensure the toggle track has a background color */
    [data-testid="stToggle"] span {
        color: #333 !important;
    }

    /* 10. Fix PyDeck Fullscreen */
    [data-testid="stDeckGlJsonChart"] {
        width: 100vw !important;
        height: 100vh !important;
    }
    [data-testid="stDeckGlJsonChart"] > div {
        width: 100% !important;
        height: 100% !important;
    }
    
    /* 11. Hide ALL Deck.gl/Mapbox UI Elements */
    [data-testid="stDeckGlJsonChart"] .mapboxgl-control-container,
    [data-testid="stDeckGlJsonChart"] .mapboxgl-ctrl,
    [data-testid="stDeckGlJsonChart"] .mapboxgl-ctrl-bottom-left,
    [data-testid="stDeckGlJsonChart"] .mapboxgl-ctrl-bottom-right,
    [data-testid="stDeckGlJsonChart"] .mapboxgl-ctrl-top-left,
    [data-testid="stDeckGlJsonChart"] .mapboxgl-ctrl-top-right,
    [data-testid="stDeckGlJsonChart"] .mapboxgl-ctrl-attrib,
    [data-testid="stDeckGlJsonChart"] .mapboxgl-compact,
    .deck-widget {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0 !important;
        width: 0 !important;
    }
    
    /* 12. Fix Slider Label Colors */
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div,
    [data-testid="stSidebar"] .stSlider p,
    [data-testid="stSidebar"] [data-baseweb="slider"] {
        color: #333 !important;
    }
    
    /* Toggle switch styling - visible grey when off */
    [data-testid="stSidebar"] .st-emotion-cache-1p2iens,
    [data-testid="stSidebar"] [role="switch"],
    [data-testid="stSidebar"] .stCheckbox > label > div[data-testid="stMarkdownContainer"] + div,
    [data-testid="stSidebar"] label[data-baseweb="checkbox"] span:first-child {
        background-color: #aaa !important;
        border: 2px solid #888 !important;
    }
    [data-testid="stSidebar"] [role="switch"][aria-checked="true"],
    [data-testid="stSidebar"] .st-emotion-cache-1p2iens[aria-checked="true"] {
        background-color: #2a9d8f !important;
        border-color: #2a9d8f !important;
    }
    /* Also target the toggle track directly */
    [data-testid="stSidebar"] div[data-baseweb="toggle"] > div:first-child {
        background-color: #aaa !important;
    }
    [data-testid="stSidebar"] div[data-baseweb="toggle"][aria-checked="true"] > div:first-child {
        background-color: #2a9d8f !important;
    }
    
    /* 13. Hide Empty Containers - Nuclear Option */
    [data-testid="stSidebar"] .element-container:empty,
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:empty,
    [data-testid="stSidebar"] > div > div:empty,
    section[data-testid="stSidebar"] div[class=""],
    section[data-testid="stSidebar"] div[style=""],
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div[data-testid="element-container"]:last-child:empty {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        visibility: hidden !important;
    }
    
    /* 14. Hide any white boxes at bottom of sidebar */
    [data-testid="stSidebar"] > div:last-child,
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > :empty {
        display: none !important;
    }
    
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data():
    import re
    from shapely.geometry import Point
    
    path = os.path.join(os.path.dirname(__file__), DATA_PATH)
    if not os.path.exists(path):
        return pd.DataFrame()
    
    try:
        df = gpd.read_parquet(path)
        
        # Extract actual coordinates from URLs (format: !3d{lat}!4d{lon})
        def extract_coords_from_url(url):
            lat_match = re.search(r'!3d([\d.]+)', str(url))
            lon_match = re.search(r'!4d([\d.]+)', str(url))
            
            if lat_match and lon_match:
                return float(lat_match.group(1)), float(lon_match.group(1))
            return None, None
        
        # Extract coordinates
        coords = df['url'].apply(extract_coords_from_url)
        df['actual_lat'], df['actual_lon'] = zip(*coords)
        
        # Update geometry with actual coordinates where available
        valid_coords = df['actual_lat'].notna()
        df.loc[valid_coords, 'geometry'] = df.loc[valid_coords].apply(
            lambda row: Point(row['actual_lon'], row['actual_lat']), axis=1
        )
        
        if df.crs != "EPSG:4326":
            df = df.to_crs("EPSG:4326")
        
        # CRITICAL: Filter out Malaysia/out-of-bounds points IMMEDIATELY
        # Singapore bounds: Lat 1.15-1.48, Lon 103.6-104.05
        df = df[
            (df.geometry.y >= 1.15) & 
            (df.geometry.y <= 1.48) &
            (df.geometry.x >= 103.6) &
            (df.geometry.x <= 104.05)
        ]
        
        # DEDUPLICATION: Keep only the record with highest review_count per restaurant name
        # This removes duplicate scrapes and ensures best data quality
        df = df.sort_values('review_count', ascending=False)
        df = df.drop_duplicates(subset=['name'], keep='first')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def get_marker_color_rgb(residual):
    # Returns [R, G, B, A]
    if residual >= 0.5:
        return [42, 157, 143, 200] # Deep Teal
    elif residual > 0.0:
        return [233, 196, 106, 200] # Saffron
    elif residual > -0.5:
        return [244, 162, 97, 200] # Sandy Brown
    else:
        return [231, 111, 81, 200] # Burnt Sienna

def main():
    # --- Floating Panel Content (Sidebar) ---
    with st.sidebar:
        # Custom Title Styling
        st.markdown("""
            <div style="margin-bottom: 12px; margin-top: 0;">
                <h1 style="font-size: 2.3rem; font-weight: 800; color: #333; margin: 0 0 0.3rem 0; line-height: 1.2;">
                    🍜 Shiok Scout
                </h1>
                <p style="font-size: 1rem; color: #555; margin: 0; font-weight: 500;">
                    Algorithmic Restaurant Gems
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # This will be updated after filtering
        visible_count_placeholder = st.empty()
        
        st.markdown("<div style='margin: 8px 0; border-top: 1px solid #e0e0e0;'></div>", unsafe_allow_html=True)

        # Load Data
        df = load_data()
        
        if df.empty:
            st.error("No data found.")
            st.stop()

        # Filters
        st.subheader("Filters")
        
        highlight_gems = st.checkbox("Highlight Gems Only", value=False)

        # Planning Area
        if 'planning_area' in df.columns:
            all_areas = ["All Areas"] + sorted(df['planning_area'].astype(str).unique().tolist())
            selected_area = st.selectbox("Planning Area", all_areas)
        else:
            selected_area = "All Areas"

        # Cuisine
        if 'category' in df.columns:
            all_cuisines = ["All Cuisines"] + sorted(df['category'].astype(str).unique().tolist())
            selected_cuisine = st.selectbox("Cuisine", all_cuisines)
        else:
            selected_cuisine = "All Cuisines"

        # Sliders
        min_rating = st.slider("Min Rating", 0.0, 5.0, 3.5, 0.1)
        min_reviews = st.slider("Min Reviews", 0, 100, 5, 5)

        st.markdown("<div style='margin: 12px 0; border-top: 1px solid #e0e0e0;'></div>", unsafe_allow_html=True)
        st.markdown("**Legend**")
        st.markdown(
            """
            <div style="font-size: 14px; color: #333;">
                <div style="display: flex; align-items: center; margin-bottom: 4px;">
                    <span style="display: inline-block; width: 10px; height: 10px; background-color: #2a9d8f; border-radius: 50%; margin-right: 8px;"></span>
                    <span>Undervalued (Gem)</span>
                </div>
                <div style="display: flex; align-items: center; margin-bottom: 4px;">
                    <span style="display: inline-block; width: 10px; height: 10px; background-color: #e9c46a; border-radius: 50%; margin-right: 8px;"></span>
                    <span>Fair Value</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <span style="display: inline-block; width: 10px; height: 10px; background-color: #e76f51; border-radius: 50%; margin-right: 8px;"></span>
                    <span>Overvalued</span>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )

    # --- Filtering Logic ---
    filtered_df = df.copy()
    
    # Note: Malaysia filtering already done in load_data()
    # Additional user-driven filters below
    
    if selected_area != "All Areas":
        filtered_df = filtered_df[filtered_df['planning_area'] == selected_area]
    
    if selected_cuisine != "All Cuisines":
        filtered_df = filtered_df[filtered_df['category'] == selected_cuisine]
        
    filtered_df = filtered_df[filtered_df['rating'] >= min_rating]
    filtered_df = filtered_df[filtered_df['review_count'] >= min_reviews]
    
    if highlight_gems:
        # Only show truly undervalued (teal dots, residual >= 0.5)
        filtered_df = filtered_df[filtered_df['residual'] >= 0.5]
    
    # Update visible count in sidebar with prominent styling
    visible_count_placeholder.markdown(
        f"""
        <div style="background: linear-gradient(90deg, #e8f5f3 0%, #f0f9f7 100%); 
                    padding: 8px 12px; 
                    border-radius: 6px; 
                    border-left: 3px solid #2a9d8f;
                    margin-bottom: 8px;">
            <span style="color: #2a9d8f; font-weight: 600; font-size: 14px;">
                📍 Showing {len(filtered_df):,} restaurants
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Prepare Data for Deck.gl ---
    # We need a clean DF with lat/lon columns and a color column
    if not filtered_df.empty:
        # Extract lat/lon directly from geometry
        # geometry.y = latitude, geometry.x = longitude
        filtered_df['lat'] = filtered_df.geometry.y
        filtered_df['lon'] = filtered_df.geometry.x
        
        # Apply color mapping
        filtered_df['color'] = filtered_df['residual'].apply(get_marker_color_rgb)
        
        # CREATE TOOLTIP COLUMNS BEFORE LAYER
        # Badge text for residual score
        filtered_df['badge'] = filtered_df['residual'].apply(
            lambda x: f"Underrated +{x:.2f}" if x >= 0 else f"Overrated {x:.2f}"
        )
        
        # Planning area for location
        filtered_df['area'] = filtered_df['planning_area'].fillna('Unknown').astype(str)
        
        # Header color matching the dot color (hex format for CSS)
        def get_header_color(residual):
            if residual >= 0.5:
                return '#2a9d8f'  # Teal - Undervalued
            elif residual > 0.0:
                return '#e9c46a'  # Saffron - Fair
            elif residual > -0.5:
                return '#f4a261'  # Sandy Brown
            else:
                return '#e76f51'  # Coral - Overvalued
        
        filtered_df['hcolor'] = filtered_df['residual'].apply(get_header_color)
        
        # Generate explanation paragraph
        def generate_explanation(row):
            predicted = row.get('predicted_rating', row['rating'] - row['residual'])
            actual = row['rating']
            residual = row['residual']
            reviews = int(row.get('review_count', 0))
            density = int(row.get('cluster_density', 0))
            area = row.get('planning_area', 'this area')
            cuisine = row.get('category', 'Other')
            
            # Build specific factor descriptions
            factors = []
            
            # Cuisine factor
            if cuisine and cuisine != 'Other':
                factors.append(f"{cuisine} cuisine")
            
            # Review factor
            if reviews < 20:
                factors.append("very few reviews")
            elif reviews < 100:
                factors.append(f"{reviews} reviews")
            elif reviews > 500:
                factors.append(f"high popularity ({reviews} reviews)")
            
            # Competition factor
            if density < 5:
                factors.append("low competition nearby")
            elif density > 40:
                factors.append("highly competitive area")
            
            factors_str = ", ".join(factors) if factors else "its profile"
            
            if residual >= 0.5:
                return f"With {factors_str}, similar spots in {area} average {predicted:.1f}★. This place beats expectations at {actual:.1f}★ - a true gem!"
            elif residual > 0.1:
                return f"Based on {factors_str}, we'd expect {predicted:.1f}★. Scoring {actual:.1f}★ means it's performing above average."
            elif residual > -0.1:
                return f"Rating of {actual:.1f}★ is in line with expectations ({predicted:.1f}★) for {factors_str}. No surprises here."
            elif residual > -0.5:
                return f"Given {factors_str}, similar places score {predicted:.1f}★. At {actual:.1f}★, this is slightly below expectations."
            else:
                return f"With {factors_str}, we'd expect {predicted:.1f}★. The {actual:.1f}★ rating suggests it's underperforming."
        
        filtered_df['explanation'] = filtered_df.apply(generate_explanation, axis=1)
        
        # Calculate View State
        if selected_area == "All Areas":
            initial_view_state = pdk.ViewState(
                latitude=1.3521,
                longitude=103.8198,
                zoom=11,
                pitch=0,
            )
        else:
            initial_view_state = pdk.ViewState(
                latitude=filtered_df['lat'].mean(),
                longitude=filtered_df['lon'].mean(),
                zoom=13,
                pitch=0,
            )
            
        # Define Layer - must come AFTER adding tooltip columns
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=filtered_df,
            get_position='[lon, lat]',
            get_fill_color='color',
            get_radius=30, # Radius in meters
            pickable=True,
            opacity=0.8,
            stroked=True,
            filled=True,
            radius_min_pixels=3,
            radius_max_pixels=10,
            line_width_min_pixels=1,
            get_line_color=[255, 255, 255, 100],
        )
        
        # Tooltip with dynamic colors
        tooltip = {
            "html": """
                <div style="font-family: -apple-system, sans-serif; width: 320px; background: white; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.2); overflow: hidden;">
                    <div style="background: {hcolor}; padding: 12px 16px;">
                        <div style="color: white; font-size: 16px; font-weight: 600;">{name}</div>
                    </div>
                    <div style="padding: 12px 16px;">
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
                            <div style="display: flex; align-items: center; gap: 4px;">
                                <span style="color: #ffa500; font-size: 18px;">★</span>
                                <span style="font-size: 15px; font-weight: 600;">{rating}</span>
                                <span style="color: #666; font-size: 12px;">({review_count})</span>
                            </div>
                            <div style="background: {hcolor}; color: white; padding: 3px 9px; border-radius: 3px; font-size: 11px; font-weight: 600;">
                                {badge}
                            </div>
                        </div>
                        <div style="color: #666; font-size: 12px; padding-top: 6px; border-top: 1px solid #eee;">
                            📍 {area}
                        </div>
                        <div style="color: #555; font-size: 11px; font-style: italic; margin-top: 8px; padding: 8px; background: #f8f8f8; border-radius: 4px; line-height: 1.4;">
                            💡 {explanation}
                        </div>
                    </div>
                </div>
            """,
            "style": {"backgroundColor": "transparent", "padding": "0"}
        }
        
        # Render Deck
        r = pdk.Deck(
            layers=[layer],
            initial_view_state=initial_view_state,
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            tooltip=tooltip,
        )
        
        # Use a dynamic key to force re-render when filters change, preventing sticky state
        map_key = f"map_{selected_area}_{highlight_gems}"
        st.pydeck_chart(r, use_container_width=True, key=map_key)
        
    else:
        st.warning("No data matches your filters.")

if __name__ == "__main__":
    main()
