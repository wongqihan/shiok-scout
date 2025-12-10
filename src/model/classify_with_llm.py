"""
Use Gemini API to classify restaurants that couldn't be categorized by keywords.
"""

import pandas as pd
import geopandas as gpd
import os
import time
import google.generativeai as genai

CUISINE_CATEGORIES = [
    'Japanese', 'Korean', 'Chinese', 'Indian', 'Thai', 'Vietnamese', 
    'Malay', 'Western', 'Italian', 'Mexican', 'Middle Eastern', 
    'Seafood', 'Hawker', 'Cafe', 'Fast Food', 'BBQ', 'Other'
]

BATCH_PROMPT = """You are a cuisine classifier for Singapore restaurants. 
Classify each restaurant into ONE of these categories:
{categories}

Rules:
- Respond with ONLY the restaurant name and category, one per line, format: "Name | Category"
- If you cannot determine the cuisine, use "Other"
- "Western" includes American, British, European, fusion restaurants
- "Hawker" is for food courts, hawker centers, eating houses
- "Cafe" is for coffee shops, bakeries, dessert places

Restaurant names:
{names}

Classifications:"""


def classify_with_llm(names: list, api_key: str, batch_size: int = 20) -> dict:
    """Classify restaurant names using Gemini API in batches."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-flash-latest')
    
    results = {}
    categories_str = ', '.join(CUISINE_CATEGORIES)
    
    # Process in batches
    for i in range(0, len(names), batch_size):
        batch = names[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(names) + batch_size - 1) // batch_size
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} restaurants)...")
        
        try:
            names_str = "\n".join(f"- {name}" for name in batch)
            prompt = BATCH_PROMPT.format(categories=categories_str, names=names_str)
            response = model.generate_content(prompt)
            
            # Parse response
            for line in response.text.strip().split("\n"):
                if "|" in line:
                    parts = line.split("|")
                    name = parts[0].strip().lstrip("- ")
                    category = parts[1].strip()
                    
                    # Validate category
                    if category not in CUISINE_CATEGORIES:
                        for cat in CUISINE_CATEGORIES:
                            if cat.lower() in category.lower():
                                category = cat
                                break
                        else:
                            category = 'Other'
                    
                    # Match to original name (fuzzy)
                    for orig_name in batch:
                        if orig_name.lower().startswith(name.lower()[:20]) or name.lower() in orig_name.lower():
                            results[orig_name] = category
                            break
            
            # Rate limiting - 10 requests per minute
            time.sleep(6)
                
        except Exception as e:
            print(f"Error processing batch: {e}")
            for name in batch:
                results[name] = 'Other'
            time.sleep(10)
    
    return results


def main():
    # Get API key from environment
    api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
    
    if not api_key:
        print("ERROR: No GOOGLE_API_KEY or GEMINI_API_KEY found in environment.")
        print("Please set: export GOOGLE_API_KEY='your-key'")
        return
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '../../data/processed/restaurants_with_residuals.parquet')
    
    print("Loading data...")
    gdf = gpd.read_parquet(data_path)
    
    # Get "Other" restaurants
    others = gdf[gdf['category'] == 'Other']
    print(f"Found {len(others)} restaurants with 'Other' category")
    
    if len(others) == 0:
        print("No 'Other' restaurants to classify!")
        return
    
    # Get unique names to classify (avoid duplicates)
    unique_names = others['name'].unique().tolist()
    print(f"Unique names to classify: {len(unique_names)}")
    
    print("\nClassifying with Gemini API...")
    classifications = classify_with_llm(unique_names, api_key)
    
    # Apply classifications
    print("\nApplying classifications...")
    gdf.loc[gdf['category'] == 'Other', 'category'] = gdf.loc[gdf['category'] == 'Other', 'name'].map(classifications)
    
    # Fill any NaN with 'Other'
    gdf['category'] = gdf['category'].fillna('Other')
    
    print("\nNew category distribution:")
    print(gdf['category'].value_counts())
    
    # Count remaining "Other"
    remaining_other = len(gdf[gdf['category'] == 'Other'])
    print(f"\nRemaining 'Other': {remaining_other} ({remaining_other/len(gdf)*100:.1f}%)")
    
    # Save
    gdf.to_parquet(data_path)
    print(f"\nSaved to {data_path}")


if __name__ == "__main__":
    main()
