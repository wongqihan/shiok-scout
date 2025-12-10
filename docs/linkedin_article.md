# How I Built "Shiok Scout": Algorithmically Finding Singapore's Hidden Gem Restaurants

*A technical walkthrough on scraping, ML residual analysis, and why the bottleneck has shifted from capital to clarity.*

---

## The Problem

Every "Best Restaurants in Singapore" list suffers from the same bias: popularity.

High-rated restaurants in Orchard attract more reviews. More reviews boost visibility. More visibility drives more reviews. It's a self-reinforcing loop that systematically overlooks great food in "ulu" locations.

I wanted to build something different: a system that finds restaurants that **outperform expectations** for their context.

Not the "best" restaurants. The **undervalued** ones.

---

## The Approach: Residual Analysis

The core insight is borrowed from finance: **alpha = actual return - expected return.**

Applied to restaurants:
- **Residual = Actual Rating - Predicted Rating**

If a restaurant has a 4.6 rating, but the model predicted 4.0 based on its location, cuisine, and review count—that's a +0.6 residual. A hidden gem.

Conversely, a 4.2-rated restaurant in a prime location with many reviews might have a *negative* residual. It's underperforming its context.

---

## Step 1: Data Collection

I scraped ~70,000 restaurant entries from Google Maps using Playwright (headless browser automation). The scraper:

- Iterated through 1,300+ geographic "seed points" across Singapore
- Extracted: name, rating, review count, coordinates, category
- Saved checkpoints every 50 seeds (crucial for resumability)

After deduplication (keeping the record with the highest review count per restaurant name), I had **3,641 unique restaurants**.

💡 *Key learning: Google Maps data is messy. Many entries had 0 reviews or generic "Restaurant" categories. Data cleaning took 40% of the effort.*

---

## Step 2: Feature Engineering

The model uses five features to predict expected rating:

| Feature | Description | Importance |
|---------|-------------|------------|
| `planning_area` | Singapore planning zone (NEWTON, ORCHARD, etc.) | **0.49** |
| `log_reviews` | Log-transformed review count | 0.36 |
| `category` | Cuisine type (Japanese, Indian, etc.) | 0.27 |
| `is_chain` | Whether the restaurant name appears 3+ times | 0.15 |
| `cluster_density` | Number of restaurants within 200m | 0.05 |

The big insight: **location dominates.** Nearly half of the rating variance is explained by *where* the restaurant is, not what it serves.

---

## Step 3: The LLM Classification Trick

Here's where modern AI tooling made a real difference.

The scraper only captured "Restaurant" as the category for most entries—not useful. I needed to classify 1,368 restaurants by cuisine type.

Old approach: Manual labeling or expensive NER APIs.

New approach: **Gemini Flash API** with batch prompting.

```python
BATCH_PROMPT = """Classify each restaurant into ONE of these categories:
Japanese, Korean, Chinese, Indian, Thai, Vietnamese, Malay, Western, 
Italian, Mexican, Middle Eastern, Seafood, Hawker, Cafe, Fast Food, BBQ, Other

Restaurant names:
{names}

Respond with: "Name | Category" for each."""
```

I sent 20 restaurants per API call, with 6-second rate limiting. Total time: **7 minutes**. Total cost: **$0.00** (within free tier).

The result: "Other" category dropped from 38% to 13%. The model now had meaningful cuisine signals.

💡 *The Shift: What would've required a contractor or manual labeling a few years ago now costs effectively nothing.*

---

## Step 4: Model Training

I used **HistGradientBoostingRegressor** from scikit-learn—a robust, fast algorithm that handles categorical features natively.

```python
model = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('cat', OrdinalEncoder(), ['category', 'planning_area']),
        ('num', 'passthrough', ['log_reviews', 'is_chain', 'cluster_density'])
    ])),
    ('regressor', HistGradientBoostingRegressor(
        categorical_features=[0, 1], 
        l2_regularization=1.0
    ))
])
```

Cross-validation RMSE: **0.42** (meaning predictions are typically within 0.4 stars of actual).

Training time: **2 seconds**.

---

## Step 5: Calculating Residuals

```python
gdf['predicted_rating'] = model.predict(X)
gdf['residual'] = gdf['rating'] - gdf['predicted_rating']
```

Restaurants with residuals ≥ 0.5 are flagged as **Hidden Gems** (teal).
Residuals between 0 and 0.5 are **Fair Value** (yellow).
Residuals below 0 are **Overvalued** (coral).

---

## Step 6: The Interactive Map

Built with **Streamlit** + **PyDeck** for GPU-accelerated map rendering.

Key features:
- Color-coded dots based on residual
- Sidebar filters: Planning Area, Cuisine, Min Rating, Min Reviews, "Gems Only" toggle
- Dynamic tooltips explaining *why* each restaurant is rated that way

Example tooltip:
> *"With Japanese cuisine, 87 reviews, similar spots in ORCHARD average 4.1★. This place beats expectations at 4.6★—a true hidden gem!"*

The explanation is generated dynamically from the model's inputs, not hardcoded.

---

## The Interesting Findings

### 🟢 Hidden Gems (Positive Residuals)

1. **Industrial areas hide gems.** The top residuals are in Sungei Kadut, Tuas, and Tengah—places most food guides ignore.

2. **Cuisine matters, but less than location.** A mediocre Japanese restaurant in Marina Bay will likely outperform a good one in Jurong, purely due to location effects.

3. **Chain restaurants are predictable.** The `is_chain` feature has low importance—chains perform almost exactly as expected.

**Top 10 Hidden Gems (10+ reviews):**

| Restaurant | Rating | Expected | Gap | Reviews | Location |
|------------|--------|----------|-----|---------|----------|
| **Chez West** | 4.9★ | 3.7★ | **+1.22** | 347 | TENGAH |
| Nonya Bong The Peranakan | 4.8★ | 3.9★ | +0.87 | 65 | SUNGEI KADUT |
| RESTAURANT SING MENG CHAI | 4.7★ | 3.9★ | +0.80 | 18 | N.E. ISLANDS |
| Italian Patio | 4.8★ | 4.1★ | +0.68 | 37 | SUNGEI KADUT |
| Sin Lam Huat | 4.9★ | 4.2★ | +0.65 | 59 | CHANGI |
| Soup Restaurant (Star Vista) | 5.0★ | 4.4★ | +0.62 | 67 | QUEENSTOWN |
| Wu You Eating Place | 4.7★ | 4.1★ | +0.61 | 318 | TUAS |
| LONG ZHUANG | 4.9★ | 4.3★ | +0.61 | 81 | CHANGI |
| Rockafellers Puteri Harbour | 4.9★ | 4.3★ | +0.60 | 591 | WESTERN CATCHMENT |
| RAJAH PRATAA | 4.6★ | 4.0★ | +0.60 | 31 | SERANGOON |

**Chez West** in Tengah remains the undisputed king of hidden gems. **Nonya Bong** in Sungei Kadut (an industrial estate) is another standout. Note that some "Western Catchment" entries might be just across the border but accessible.

### 🔴 The Overhyped List (Negative Residuals)

I also found restaurants that are **underperforming** their potential—popular spots with hundreds of reviews, but ratings significantly below what they should achieve given their context.

**Top 10 Overhyped Restaurants (100+ reviews):**

| Restaurant | Rating | Expected | Gap | Reviews |
|------------|--------|----------|-----|---------|
| Domino's Pizza Choa Chu Kang | 2.1★ | 3.5★ | **-1.44** | 354 |
| Marine Beach Bar | 2.8★ | 4.0★ | -1.24 | 207 |
| My Briyani House | 2.9★ | 4.0★ | -1.12 | 458 |
| Sanchon Korean Cuisine | 2.7★ | 3.8★ | -1.05 | 303 |
| D'Rubinah Eating Place | 3.2★ | 4.2★ | -1.02 | 176 |
| Woody Family Peranakan Cafe | 3.5★ | 4.5★ | -0.96 | 1,038 |
| **Thohirah Restaurant** | 3.3★ | 4.2★ | -0.90 | **4,249** |
| Season Live Seafood | 2.9★ | 3.8★ | -0.88 | 333 |
| MR. PRATA | 3.2★ | 4.0★ | -0.84 | 1,086 |
| Fusion Spoon | 3.4★ | 4.1★ | -0.74 | 690 |

**Thohirah Restaurant** stands out: over 4,000 reviews but rating 0.9★ below expected. That's a lot of disappointed customers for a spot that should be performing better given its category and location.

Common patterns in overhyped spots:
- **Fast food chains** known for inconsistency
- **Tourist-heavy locations** with captive audiences
- **High-volume spots** where service quality suffers

---

## The Unit Economics of Weekend Projects

Let me break down the actual costs:

| Resource | Cost |
|----------|------|
| Scraping (Playwright on local machine) | $0 |
| Gemini API (1,368 classifications) | $0 |
| Cloud hosting | $0 (Streamlit Cloud free tier) |
| Model training | $0 (CPU, 2 seconds) |
| **Total** | **$0** |

Time investment: ~12 hours over a weekend.

Five years ago, this project would've required licensed data, cloud compute, and probably a team. Today, the marginal cost is zero. The only bottleneck is *having the idea* and *knowing what question to ask.*

---

## Reflections

The most valuable part of this project wasn't the map. It was reframing the question.

"What's the best restaurant?" is a popularity contest.
"What's *surprisingly good* for its context?" is an insight engine.

The same reframe applies everywhere:
- Not "what's the top-performing ad?" but "what's overperforming given its budget?"
- Not "who's the best hire?" but "who's outperforming expectations given their background?"

The tools are nearly free now. The skill is asking better questions.

---

**🍜 Try Shiok Scout: https://shiok-scout.streamlit.app**

---

## Image Prompt (for Article Header)

"A conceptual illustration of a magnifying glass hovering over a stylized map of Singapore. Through the magnifying glass, certain dots on the map glow bright teal while others remain muted amber and coral. The lens creates a 'reveal' effect—hidden value being uncovered. The aesthetic is clean, data-driven, and slightly futuristic. Dark background with subtle grid lines. No text, no faces, no photographs. Professional and abstract."
