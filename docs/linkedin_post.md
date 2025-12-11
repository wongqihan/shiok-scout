# LinkedIn Post: Shiok Scout

**I built a "hidden gem" restaurant finder for Singapore in a weekend. Here's what I learned:**

The idea was simple: can we algorithmically find undervalued restaurants—places that punch above their weight for their context?

Not just "high ratings," but places that *overperform* given their location, cuisine type, and popularity.

💡 **The Shift I Keep Seeing**

A few years ago, this project would've required:
- Licensing expensive location data
- A data science team to build the ML pipeline
- Weeks of development

Last weekend, I did it with:
- Free scraped data (Playwright + Python)
- Gemini API to classify 1,300+ restaurants by cuisine ($0.00 - within free tier)
- A gradient boosting model that took 2 seconds to train

The marginal cost of execution has collapsed. The bottleneck is no longer capital—it's clarity of the question.

📊 **The Interesting Findings**

The model revealed that **location is king**. Planning area alone explains ~49% of rating variance. Cuisine type accounts for ~27%.

## The Problem: The "Best" List is Broken.

Every "Best Restaurants" list suffers from the same bias: **digital visibility**.

High-rated spots in prime locations attract more visits. More visits = more reviews. More reviews = higher visibility. It's a flywheel that systematically overlooks great food in "ulu" locations.

I wanted to build something different: a system that strips away the location bias to find restaurants that are **statistically overperforming** their context.

**A few caveats for V1:**
1. **No Hawkers:** I deliberately excluded hawker centers—they operate on a different scale and deserve their own model.
2. **The "Free Dessert" Problem:** Reviews can be inflated by promos ("5 stars for free ice cream!") or bots. Detecting this requires analyzing review content, which is hard to do at scale without aggressive scraping. For now, treat high residuals as a signal to investigate, not a guarantee.

**Easter Egg:** The scraper accidentally (but happily) captured parts of **Johor Bahru**—so you can find gems for your weekend trips too. 🚗💨

## Top Findings

1.  **Industrial Estates are Gold Mines.**
    Forget Orchard. The highest "Value Above Replacement" scores are consistently found in **Sungei Kadut, Tuas, and Tengah**. These spots survive purely on food quality, with zero foot traffic to save them.

2.  **The "Marina Bay Premium" is Real.**
    Location explains nearly 50% of a rating. A mediocre Japanese restaurant in Marina Bay will statistically drift towards 4.2 stars, while a superior one in Jurong fights to hit 4.0. The model exposes this gap.

3.  **Chains are the Control Group.**
    Chain restaurants (McDonald's, Saizeriya, etc.) have near-zero residuals. They perform exactly as the model predicts—perfectly average, perfectly predictable. They are the baseline against which "Gems" are measured.

### Top 10 Gems (Undervalued, >10 Reviews)
| Restaurant | Location | Google Rating | Model Rating | Gap | Reviews |
| --- | --- | --- | --- | --- | --- |
| Chez West | TENGAH | 4.9 | 3.70 | +1.20 | 347 |
| Nonya Bong The Peranakan | SUNGEI KADUT | 4.8 | 3.89 | +0.91 | 65 |
| Italian Patio | SUNGEI KADUT | 4.8 | 4.13 | +0.67 | 37 |
| The Hidden Bar and Bistro | LIM CHU KANG | 5.0 | 4.36 | +0.64 | 13 |
| Wok of Home Town | BUKIT BATOK | 5.0 | 4.36 | +0.64 | 17 |
| AL MAAS RESTAURANT | TOA PAYOH | 4.7 | 4.08 | +0.62 | 46 |
| Burger King SAFRA Choa Chu Kang | CHOA CHU KANG | 4.9 | 4.30 | +0.60 | 1214 |
| D'Penyetz Hillion Mall | CENTRAL WATER CATCHMENT | 4.8 | 4.20 | +0.60 | 145 |
| Little Pond 小鱼塘@West Mall | BUKIT BATOK | 4.9 | 4.31 | +0.59 | 1053 |
| The Flying Chef Private Kitchen | BEDOK | 5.0 | 4.41 | +0.59 | 71 |

### Bottom 10 (Overvalued, >10 Reviews)
| Restaurant | Location | Google Rating | Model Rating | Gap | Reviews |
| --- | --- | --- | --- | --- | --- |
| Ming Kitchen Seafood | PASIR RIS | 1.6 | 3.63 | -2.03 | 58 |
| Kunyit Kedai | SUNGEI KADUT | 1.7 | 3.23 | -1.53 | 43 |
| The Terrace | QUEENSTOWN | 2.2 | 3.69 | -1.49 | 11 |
| Domino's Pizza Choa Chu Kang | TENGAH | 2.1 | 3.53 | -1.43 | 354 |
| Boon Lay Cafe | BOON LAY | 1.8 | 2.99 | -1.19 | 67 |
| Sanchon Korean Cuisine | CHANGI | 2.7 | 3.85 | -1.15 | 303 |
| Wen Jia Bao | BOON LAY | 1.6 | 2.73 | -1.13 | 68 |
| Hong Kong Sheng Kee Noodle House | CHANGI | 2.9 | 3.97 | -1.07 | 67 |
| My Briyani House | BEDOK | 2.9 | 3.90 | -1.00 | 458 |
| The Queen’s Grill | HOUGANG | 3.2 | 4.19 | -0.99 | 87 |

---

**What undervalued opportunities are hiding in your industry's data?**

🔗 Try it: https://shiok-scout.streamlit.app

#DataScience #Singapore #AI #ProductDevelopment

---

## Image Prompt

"A stylized top-down map of Singapore rendered as a clean data visualization. Scattered across the map are glowing dots in three colors: teal, amber, and coral. The teal dots pulse slightly brighter, representing hidden gems. The background is a subtle dark grid pattern with soft gradients. The aesthetic is modern, minimal, and professional—like a Bloomberg terminal crossed with a food delivery app. No text, no faces, no photographs."
