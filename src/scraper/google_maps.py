from playwright.sync_api import sync_playwright
import time
import random
import re
import numpy as np
from typing import List, Dict, Optional

class RestaurantScraper:
    def __init__(self, headless: bool = True, use_dummy: bool = False):
        self.headless = headless
        self.use_dummy = use_dummy
        self.playwright = None
        self.browser = None
        
        if not self.use_dummy:
            print("Initialized RestaurantScraper in REAL mode (Playwright).")
            self.playwright = sync_playwright().start()
            try:
                # Try using the bundled chromium first
                self.browser = self.playwright.chromium.launch(headless=self.headless)
            except Exception as e:
                print(f"Bundled Chromium not found ({e}). Trying system Chrome...")
                try:
                    self.browser = self.playwright.chromium.launch(headless=self.headless, channel="chrome")
                except Exception as e2:
                    print(f"System Chrome not found either ({e2}). Trying Edge...")
                    self.browser = self.playwright.chromium.launch(headless=self.headless, channel="msedge")
        else:
            print("Initialized RestaurantScraper in DUMMY mode.")
        
    def scrape_location(self, lat: float, lon: float, radius_meters: int = 500) -> List[Dict]:
        """
        Scrapes restaurants near a location using Google Maps.
        """
        if self.use_dummy:
            return self._generate_dummy_data(lat, lon)
        
        results = []
        context = self.browser.new_context(
            viewport={'width': 1280, 'height': 800},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        page = context.new_page()
        
        try:
            # Search for "restaurants" at the location
            url = f"https://www.google.com/maps/search/restaurants/@{lat},{lon},16z"
            print(f"Navigating to {url}...")
            page.goto(url, timeout=60000)
            
            # Wait for the feed to load
            try:
                page.wait_for_selector('div[role="feed"]', timeout=10000)
            except:
                print("Feed not found, possibly no results or different layout.")
                # Sometimes it shows a list directly or just pins.
                # Try checking for 'div[role="article"]' directly
                pass

            # Scroll the feed to load more results
            # The feed is usually the div with role="feed"
            feed_selector = 'div[role="feed"]'
            
            # Scroll a few times
            for _ in range(3): 
                page.keyboard.press("PageDown")
                time.sleep(1)
                # Also try JS scroll
                page.evaluate(f"""
                    const feed = document.querySelector('{feed_selector}');
                    if (feed) {{
                        feed.scrollTop = feed.scrollHeight;
                    }}
                """)
                time.sleep(2)

            # Extract items
            # Items are usually links with class 'hfpxzc' (the invisible link covering the card)
            # OR we can look for the card containers.
            # A robust way is to find all elements with an aria-label that contains "stars" or look for the main card structure.
            
            # Let's try to find the main card containers.
            # They are usually direct children of the feed's internal div.
            # But relying on 'hfpxzc' class which is the "Link to place" is fairly common in recent GMaps.
            
            # Strategy: Find all links that look like place links
            place_links = page.locator('a[href^="https://www.google.com/maps/place"]').all()
            
            print(f"Found {len(place_links)} potential spots. Parsing...")
            
            for link in place_links:
                try:
                    # The aria-label of the link is usually the Name
                    name = link.get_attribute("aria-label")
                    if not name:
                        continue
                        
                    url = link.get_attribute("href")
                    
                    # To get rating and other details, we need to look at the siblings or parent's text content.
                    # The link is usually an overlay. The text is "behind" it in the DOM structure (siblings).
                    # Actually, the link is often inside a div that has siblings with the text.
                    # Let's try to get the parent container of the link and extract text from there.
                    
                    # In the sidebar list, the structure is roughly:
                    # Container
                    #   - Link (overlay) -> has Name in aria-label
                    #   - Div (Text info)
                    #     - Div (Rating + Stars)
                    #     - Div (Category + Price)
                    
                    # We can get the parent of the link
                    card_container = link.locator("..") 
                    
                    # Get all text from the card
                    text_content = card_container.inner_text()
                    lines = text_content.split('\n')
                    
                    # Parse Rating and Reviews
                    # Look for pattern like "4.5(1,200)" or "4.5" and stars
                    rating = None
                    review_count = 0
                    
                    # Usually the rating is in a span with role="img" aria-label="4.5 stars 1,200 Reviews"
                    # Let's try to find that specific element within the card container
                    star_el = card_container.locator('span[role="img"][aria-label*="stars"]')
                    if star_el.count() > 0:
                        aria = star_el.first.get_attribute("aria-label")
                        # Format: "4.5 stars 1,200 Reviews"
                        match = re.search(r'(\d+\.\d+)\s+stars\s+([\d,]+)\s+Reviews', aria, re.IGNORECASE)
                        if match:
                            rating = float(match.group(1))
                            review_count = int(match.group(2).replace(',', ''))
                        else:
                            # Try simpler format
                            match_rating = re.search(r'(\d+\.\d+)\s+stars', aria)
                            if match_rating:
                                rating = float(match_rating.group(1))
                    
                    # If regex failed, try parsing text lines
                    if rating is None:
                        for line in lines:
                            if re.match(r'^\d+\.\d+$', line.strip()): # Just "4.5"
                                rating = float(line.strip())
                                break
                    
                    # Price Level and Category
                    price_level = None
                    category = "Unknown"
                    
                    # Category is often the line after rating or part of the metadata line
                    # Price level is usually $, $$, etc.
                    for line in lines:
                        if line.count('$') > 0 and len(line.strip()) <= 5: # $, $$, $$$$
                             price_level = line.strip()
                    
                    # Category is harder to pinpoint without specific classes. 
                    # It's usually one of the text lines.
                    # Let's take the second line if available, or just default.
                    if len(lines) > 1:
                        # Heuristic: Category often contains "Restaurant", "Cafe", "Hawker", etc.
                        # Or it's just the line that isn't the name, rating, or status (Open/Closed).
                        pass
                        
                    # Clean up
                    if not price_level:
                        price_level = '$' # Default
                    
                    # Add to results
                    if rating:
                        results.append({
                            'name': name,
                            'address': "Singapore", # Placeholder, hard to get exact address from list view easily
                            'latitude': lat, # Approx, or extract from URL if possible
                            'longitude': lon,
                            'rating': rating,
                            'review_count': review_count,
                            'price_level': price_level,
                            'category': 'Restaurant', # Placeholder
                            'url': url
                        })
                        
                except Exception as e:
                    # print(f"Error parsing item: {e}")
                    continue
                    
        except Exception as e:
            print(f"Scraping error: {e}")
        finally:
            context.close()
            
        print(f"Scraped {len(results)} items.")
        return results

    def _generate_dummy_data(self, lat: float, lon: float) -> List[Dict]:
        """Generates fake restaurant data for testing the pipeline."""
        # Simulate network delay
        time.sleep(0.01)
        
        num_spots = random.randint(3, 15)
        results = []
        
        prefixes = ['Ah', 'Uncle', 'Grandma', 'Best', 'Tasty', 'Singapore', 'Golden', 'Silver', 'Happy', 'Lucky']
        suffixes = ['Huat', 'Seng', 'Kee', 'Kitchen', 'Bistro', 'Cafe', 'Restaurant', 'Stall', 'Delights', 'Eats']
        cuisines = ['Chicken Rice', 'Laksa', 'Western', 'Japanese', 'Indian', 'Malay', 'Chinese', 'Cafe', 'Fast Food', 'Thai', 'Seafood', 'Noodles']
        
        for _ in range(num_spots):
            # Perturb location slightly (approx within 200m)
            # 0.002 degrees is approx 220m
            r_lat = lat + random.uniform(-0.002, 0.002)
            r_lon = lon + random.uniform(-0.002, 0.002)
            
            cuisine = random.choice(cuisines)
            name = f"{random.choice(prefixes)} {random.choice(suffixes)} {cuisine}"
            
            # Generate rating (skewed towards 4.0)
            rating = round(np.clip(random.gauss(4.0, 0.5), 1.0, 5.0), 1)
            
            # Review count (Log-normal distribution)
            # Most places have few reviews, some have many
            review_count = int(np.expm1(random.uniform(1, 6))) # 2 to ~400
            if random.random() < 0.1:
                review_count = int(np.expm1(random.uniform(6, 9))) # Occasional viral spot
            
            price_level = random.choices(['$', '$$', '$$$', '$$$$'], weights=[0.5, 0.3, 0.15, 0.05])[0]
            
            results.append({
                'name': name,
                'address': f"{random.randint(1, 999)} Dummy Street, Singapore {random.randint(100000, 800000)}",
                'latitude': r_lat,
                'longitude': r_lon,
                'rating': rating,
                'review_count': review_count,
                'price_level': price_level,
                'category': cuisine,
                'url': f"https://maps.google.com/?q={r_lat},{r_lon}"
            })
            
        return results

    def close(self):
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
