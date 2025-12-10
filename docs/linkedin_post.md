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

The top "hidden gems"? Mostly in industrial areas like Sungei Kadut and Tuas. Places you'd never find on a "Top 10" list—but statistically, they're outperforming similar restaurants by 0.9+ stars.

🔥 **The Overhyped List**

On the flip side, I found restaurants that are *underperforming* their potential. Popular spots with hundreds of reviews—but ratings well below what you'd expect.

The most striking: **Thohirah Restaurant** in Sengkang. 4,249 reviews but rating 0.9★ below its predicted score. That's a lot of disappointed customers for a spot that should be doing better.

🍜 **The Product: Shiok Scout**

An interactive map of Singapore. Color-coded dots show which restaurants are undervalued (teal), fairly valued (yellow), or overvalued (coral). Each tooltip explains *why*—based on the restaurant's cuisine, review count, and competitive density.

The real unlock wasn't the tech. It was reframing the question from "what's the best rated?" to "what's surprisingly good given its circumstances?"

---

**What undervalued opportunities are hiding in your industry's data?**

🔗 Try it: https://shiok-scout.streamlit.app

#DataScience #Singapore #AI #ProductDevelopment

---

## Image Prompt

"A stylized top-down map of Singapore rendered as a clean data visualization. Scattered across the map are glowing dots in three colors: teal, amber, and coral. The teal dots pulse slightly brighter, representing hidden gems. The background is a subtle dark grid pattern with soft gradients. The aesthetic is modern, minimal, and professional—like a Bloomberg terminal crossed with a food delivery app. No text, no faces, no photographs."
