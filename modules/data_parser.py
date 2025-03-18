import requests
from bs4 import BeautifulSoup
import logging

def fetch_and_parse_article(url: str, max_length: int = 5000) -> str:
    """
    Fetch and parse article content
    
    Args:
        url (str): Article URL
        max_length (int): Maximum content length
    
    Returns:
        Extracted article text
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script, style, and navigation elements
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()
        
        # Extract text
        text = soup.get_text(separator=' ', strip=True)
        
        # Truncate to max length
        return text[:max_length]
    
    except Exception as e:
        logging.error(f"Article parsing error for {url}: {e}")
        return ""