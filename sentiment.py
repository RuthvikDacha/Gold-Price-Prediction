# sentiment.py
# News sentiment analysis using VADER and Yahoo Finance news headlines.
#
# Important note about what this does and doesn't do:
# yfinance only provides the last few days of headlines, which means I can't
# build a historical sentiment dataset long enough to train the model on.
# So sentiment is used as a supplementary signal — it shows up on the dashboard
# to give context around the current prediction, not as a model feature.
#
# VADER (Valence Aware Dictionary and sEntiment Reasoner) is well-suited here
# because it was specifically designed for short, informal financial/social text.
# It doesn't need training and runs instantly with no API costs.

import yfinance as yf
from datetime import datetime

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False


def get_gold_news_sentiment(max_articles: int = 10) -> dict:
    """
    Fetches recent gold-related headlines via yfinance and scores each one
    using VADER sentiment analysis.

    VADER compound score ranges:
      > 0.05  → Positive
      < -0.05 → Negative
      else    → Neutral

    Returns a dict with:
        articles      — list of headline dicts with title, sentiment, score, source, link
        avg_compound  — average sentiment score across all articles (-1 to +1)
        overall       — "Positive", "Negative", or "Neutral"
        available     — False if VADER isn't installed
        article_count — number of articles found
    """
    if not VADER_AVAILABLE:
        return {
            "available":     False,
            "articles":      [],
            "avg_compound":  0.0,
            "overall":       "Unavailable",
            "article_count": 0,
        }

    try:
        analyzer = SentimentIntensityAnalyzer()
        ticker   = yf.Ticker("GC=F")
        raw_news = ticker.news or []

        articles = []
        for item in raw_news[:max_articles]:
            # yfinance news structure can vary — handle both old and new formats
            if isinstance(item, dict):
                content  = item.get("content", item)   # newer yfinance nests under "content"
                title    = (
                    content.get("title", "")
                    or item.get("title", "")
                )
                link     = (
                    content.get("canonicalUrl", {}).get("url", "")
                    or item.get("link", "")
                    or item.get("url", "")
                )
                provider = (
                    content.get("provider", {}).get("displayName", "")
                    or item.get("publisher", "")
                    or "Unknown"
                )
            else:
                continue

            if not title:
                continue

            scores   = analyzer.polarity_scores(title)
            compound = scores["compound"]

            if compound > 0.05:
                sentiment = "Positive"
                emoji     = "📈"
            elif compound < -0.05:
                sentiment = "Negative"
                emoji     = "📉"
            else:
                sentiment = "Neutral"
                emoji     = "➡️"

            articles.append({
                "title":     title,
                "sentiment": sentiment,
                "emoji":     emoji,
                "compound":  round(compound, 3),
                "positive":  round(scores["pos"], 3),
                "negative":  round(scores["neg"], 3),
                "neutral":   round(scores["neu"], 3),
                "source":    provider,
                "link":      link,
            })

        if not articles:
            return {
                "available":     True,
                "articles":      [],
                "avg_compound":  0.0,
                "overall":       "No Data",
                "article_count": 0,
            }

        avg = sum(a["compound"] for a in articles) / len(articles)

        if avg > 0.05:
            overall = "Positive"
        elif avg < -0.05:
            overall = "Negative"
        else:
            overall = "Neutral"

        return {
            "available":     True,
            "articles":      articles,
            "avg_compound":  round(avg, 3),
            "overall":       overall,
            "article_count": len(articles),
        }

    except Exception as e:
        print(f"Sentiment fetch error: {e}")
        return {
            "available":     True,
            "articles":      [],
            "avg_compound":  0.0,
            "overall":       "Error",
            "article_count": 0,
        }
