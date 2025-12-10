"""
Deduce cuisine type from restaurant names using keyword matching.
This script updates the 'category' field based on restaurant name patterns.
"""

import pandas as pd
import geopandas as gpd
import re
import os

# Cuisine keywords mapping
CUISINE_KEYWORDS = {
    'Japanese': [
        'sushi', 'ramen', 'izakaya', 'japanese', 'nihon', 'sakura', 'tokyo', 'osaka', 
        'tempura', 'udon', 'soba', 'yakiniku', 'yakitori', 'donburi', 'bento', 'matcha',
        'miso', 'teriyaki', 'tonkatsu', 'gyudon', 'ichiban', 'nippon', 'teppanyaki',
        'omakase', 'kaiseki', 'unagi', 'takoyaki', 'okonomiyaki', 'shabu', 'tamago',
        'wagyu', 'hokkaido', 'kyoto', 'miyuki', 'kura', 'iki', 'yaki', 'katsu',
        'unatoto', 'koma', 'kanpai'
    ],
    'Korean': [
        'korean', 'korea', 'kimchi', 'bibimbap', 'bulgogi', 'kbbq', 'soju', 'seoul',
        'gogi', 'samgyeopsal', 'jjigae', 'tteok', 'banchan', 'galbi', 'anju', 'ramyeon',
        'chimaek', 'dakgalbi', 'hanwoo', 'pojangmacha', 'haedeum', 'kotuwa',
        'ajumma', 'seorae', 'jib'
    ],
    'Chinese': [
        'chinese', 'dim sum', 'dumpling', 'canton', 'szechuan', 'sichuan', 'peking',
        'hunan', 'teochew', 'hokkien', 'hainanese', 'hong kong', 'wok', 'noodle',
        'congee', 'bak kut teh', 'char siu', 'chilli crab', 'claypot', 'zi char',
        'coffee shop', 'kopitiam', '餐厅', '海鲜', '酒家', '饭店', '小吃', 'crystal jade',
        'din tai fung', 'paradise', 'canton', 'imperial', 'dragon', 'golden', 'jade',
        'oriental', 'lucky', 'prosperity', 'nanyang', 'swee', 'kee', 'heng',
        'hot pot', 'hotpot', 'steamboat', 'haidilao', 'tanyu', 'putien', 'peach garden',
        'white restaurant', 'porridge', 'frog', '农耕记', '湖南', '火锅', '川', 'ipoh',
        'tim ho wan', 'yum cha', 'roast', 'goose', 'duck', 'blossom', 'happy lamb'
    ],
    'Indian': [
        'indian', 'india', 'curry', 'tandoori', 'biryani', 'naan', 'masala', 'dosa',
        'thali', 'punjabi', 'mughlai', 'hyderabadi', 'chennai', 'madras', 'kerala',
        'tikka', 'samosa', 'paneer', 'dal', 'lassi', 'chapati', 'roti prata', 'murtabak',
        'banana leaf', 'vegetarian', 'saravana', 'komala', 'ananda', 'bhavan', 'krishna',
        'muthu', 'thanjai', 'anna', 'unavagam', 'tiffin', 'prata', 'mtr', 'gayatri',
        'saffron'
    ],
    'Thai': [
        'thai', 'thailand', 'tom yum', 'pad thai', 'green curry', 'basil', 'bangkok',
        'phuket', 'somtam', 'papaya', 'satay', 'mookata', 'boat noodle', 'lemongrass',
        'coconut', 'sticky rice', 'nana', 'siam', 'sawadee', 'sabai', 'krua', 'galangal'
    ],
    'Vietnamese': [
        'vietnamese', 'vietnam', 'pho', 'banh mi', 'bun', 'spring roll', 'hanoi',
        'saigon', 'com', 'goi cuon', 'ca phe', 'nuoc', 'nem', 'lau'
    ],
    'Malay': [
        'malay', 'nasi lemak', 'rendang', 'satay', 'laksa', 'mee goreng', 'nasi goreng',
        'rojak', 'otah', 'ayam', 'ikan', 'sambal', 'belacan', 'kampung', 'warung',
        'gerai', 'mamak', 'penang', 'melaka', 'johor', 'kedai', 'restoran',
        'mandi', 'al-azhar', 'al-ameen', 'thohirah', 'rahmat', 'siakap', 'al makan'
    ],
    'Western': [
        'western', 'grill', 'steakhouse', 'steak', 'burger', 'pasta', 'pizza', 
        'bistro', 'brasserie', 'cafe', 'coffee', 'brunch', 'breakfast', 'diner',
        'american', 'british', 'european', 'french', 'italian', 'spanish', 'german',
        'mediterranean', 'bar & grill', 'pub', 'tavern', 'gastropub', 'wine',
        'artisan', 'kitchen', 'eatery', 'provisions', 'canteen', 'deli',
        'poulet', 'marche', 'marché', 'colony', 'estate', 'club', 'pool', 'beach',
        'arbora', 'atlas', 'fico', 'ikea', 'whisk', 'paddle', 'orto', 'boiler',
        'paulaner', 'bräuhaus', 'picanhas', 'braseiro', 'chimichanga', 'supper'
    ],
    'Italian': [
        'italian', 'italy', 'pasta', 'pizza', 'risotto', 'trattoria', 'osteria',
        'ristorante', 'pizzeria', 'gelato', 'espresso', 'romano', 'napoli', 'milano',
        'venetian', 'tuscan', 'sicilian', 'parma', 'bella', 'buona', 'rosso vino'
    ],
    'Mexican': [
        'mexican', 'mexico', 'taco', 'burrito', 'quesadilla', 'nachos', 'guacamole',
        'salsa', 'enchilada', 'fajita', 'cantina', 'el ', 'la ', 'los ', 'latin',
        'guzman', 'gomez'
    ],
    'Middle Eastern': [
        'turkish', 'lebanese', 'arab', 'arabic', 'middle east', 'kebab', 'shawarma',
        'falafel', 'hummus', 'pita', 'meze', 'mezze', 'anatolia', 'ayasofya', 'ottoman',
        'persian', 'iranian', 'afghan', 'mediterranean', 'greek', 'kouzina', 'blu kouzina'
    ],
    'Seafood': [
        'seafood', 'fish', 'crab', 'lobster', 'prawn', 'shrimp', 'oyster', 'clam',
        'mussel', 'squid', 'octopus', 'salmon', 'tuna', 'sea ', 'ocean', 'marine',
        'harbour', 'pier', 'catch', 'net', 'coastes'
    ],
    'Hawker': [
        'hawker', 'food court', 'food centre', 'kopitiam', 'coffee shop', 'zi char',
        'eating house', 'market', 'food hall', 'food village', 'settlement'
    ],
    'Cafe': [
        'cafe', 'café', 'bakery', 'patisserie', 'dessert', 'cake', 'tea house',
        'bubble tea', 'boba', 'toast', 'seats'
    ],
    'Fast Food': [
        "mcdonald", 'kfc', 'burger king', 'subway', 'popeyes', 'jollibee', 'wendy',
        'taco bell', 'five guys', 'shake shack', 'carl', 'in-n-out', 'chick-fil',
        'texas chicken'
    ],
    'BBQ': [
        'bbq', 'barbecue', 'charcoal', 'smokehouse', 'fire pit', 'pitz'
    ]
}

def deduce_cuisine(name: str) -> str:
    """Deduce cuisine type from restaurant name."""
    if not name or pd.isna(name):
        return 'Other'
    
    name_lower = name.lower()
    
    # Check each cuisine's keywords
    scores = {}
    for cuisine, keywords in CUISINE_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in name_lower:
                # Longer keywords get higher scores
                score += len(keyword)
        if score > 0:
            scores[cuisine] = score
    
    # Return cuisine with highest score
    if scores:
        return max(scores, key=scores.get)
    
    return 'Other'


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '../../data/processed/restaurants_with_residuals.parquet')
    
    print("Loading data...")
    gdf = gpd.read_parquet(data_path)
    
    print(f"Total restaurants before dedup: {len(gdf)}")
    
    # Deduplicate: keep record with highest review_count per name
    print("Deduplicating...")
    gdf = gdf.sort_values('review_count', ascending=False)
    gdf = gdf.drop_duplicates(subset=['name'], keep='first')
    print(f"Total restaurants after dedup: {len(gdf)}")
    
    print(f"Current categories: {gdf['category'].value_counts().head()}")
    
    print("\nDeducing cuisines from names...")
    gdf['category'] = gdf['name'].apply(deduce_cuisine)
    
    print("\nNew category distribution:")
    print(gdf['category'].value_counts())
    
    # Save updated data
    output_path = data_path  # Overwrite
    gdf.to_parquet(output_path)
    print(f"\nSaved updated data to {output_path}")
    
    # Show some examples
    print("\nSample deductions:")
    samples = gdf.sample(20, random_state=42)[['name', 'category']]
    for _, row in samples.iterrows():
        print(f"  {row['name'][:40]:40} -> {row['category']}")


if __name__ == "__main__":
    main()
