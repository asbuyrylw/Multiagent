
import requests

def search_web(query):
    # Dummy implementation - in production, use SerpAPI or Bing Search API
    return [
        f"https://example.com/search?q={query.replace(' ', '+')}&page=1",
        f"https://example.com/search?q={query.replace(' ', '+')}&page=2"
    ]
