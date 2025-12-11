**Restaurant ratings are broken because location is a cheat code.**

A mediocre 4.0-star cafe in Orchard gets traffic because it's in Orchard. A 4.0-star cafe in Tuas has to fight for every customer. They are not the same.

I scraped 70,000+ Google Maps listings to quantify this. The data shows that **location alone explains ~50% of a restaurant's rating**.

So I built a tool to strip that bias away.

**Shiok Scout** uses a gradient boosting model to predict what a restaurant's rating *should* be, based on its location, cuisine, and density. Then it calculates the difference.

- **Positive Residual:** The restaurant is overperforming its context (a true gem).
- **Negative Residual:** It's underperforming (coasting on location).

The result is a map of Singapore that highlights "Value Above Replacement"—places that are statistically better than they have any right to be.

It’s live here: https://shiok-scout.streamlit.app

#DataScience #Singapore #FoodTech
