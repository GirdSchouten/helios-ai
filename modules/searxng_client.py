# modules/searxng_client.py
import requests
import random
import logging
from urllib.parse import urlencode

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearXNGClient:
    def __init__(self, instance_url="http://localhost:8080"):
        self.instance_url = instance_url
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
        ]
        self.session = requests.Session()
    
    def _get_headers(self):
        """Generate dynamic headers"""
        return {
            "User-Agent": random.choice(self.user_agents),
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": self.instance_url,
            "Origin": self.instance_url
        }
    
    def search(self, query, num_results=5, categories=None):
        """
        Advanced search with multiple error handling strategies
        
        Args:
            query (str): Search query
            num_results (int): Number of results
            categories (list): Search categories
        
        Returns:
            List of search results
        """
        search_strategies = [
            self._search_standard,
            self._search_with_encoded_query,
            self._search_with_alternative_endpoint
        ]
        
        for strategy in search_strategies:
            try:
                results = strategy(query, num_results, categories)
                if results:
                    return results
            except Exception as e:
                logger.warning(f"Strategy {strategy.__name__} failed: {e}")
        
        logger.error("All search strategies failed")
        return []
    
    def _search_standard(self, query, num_results, categories):
        """Standard search method"""
        params = {
            "q": query,
            "format": "json",
            "categories": categories or "news",
            "count": num_results,
        }
        
        url = f"{self.instance_url}/search"
        
        response = self.session.get(
            url, 
            params=params, 
            headers=self._get_headers(),
            timeout=10
        )
        
        response.raise_for_status()
        results = response.json().get("results", [])
        
        logger.info(f"Standard search successful: {len(results)} results")
        return results[:num_results]
    
    def _search_with_encoded_query(self, query, num_results, categories):
        """Alternative search with URL encoding"""
        url = f"{self.instance_url}/search"
        
        # Manual URL encoding
        encoded_params = urlencode({
            "q": query,
            "format": "json",
            "categories": categories or "news",
            "count": num_results,
        })
        
        full_url = f"{url}?{encoded_params}"
        
        response = self.session.get(
            full_url, 
            headers=self._get_headers(),
            timeout=10
        )
        
        response.raise_for_status()
        results = response.json().get("results", [])
        
        logger.info(f"Encoded query search successful: {len(results)} results")
        return results[:num_results]
    
    def _search_with_alternative_endpoint(self, query, num_results, categories):
        """Try alternative SearXNG endpoints"""
        alternative_endpoints = [
            "/search",
            "/api/search",
            "/json"
        ]
        
        for endpoint in alternative_endpoints:
            try:
                params = {
                    "q": query,
                    "format": "json",
                    "categories": categories or "news",
                    "count": num_results,
                }
                
                url = f"{self.instance_url}{endpoint}"
                
                response = self.session.get(
                    url, 
                    params=params, 
                    headers=self._get_headers(),
                    timeout=10
                )
                
                response.raise_for_status()
                results = response.json().get("results", [])
                
                if results:
                    logger.info(f"Alternative endpoint {endpoint} successful: {len(results)} results")
                    return results[:num_results]
            
            except Exception as e:
                logger.warning(f"Alternative endpoint {endpoint} failed: {e}")
        
        raise ValueError("No successful alternative endpoints found")
    
    def diagnose_connection(self):
        """Comprehensive connection diagnosis"""
        try:
            # Check base URL accessibility
            base_response = requests.get(self.instance_url, timeout=5)
            base_response.raise_for_status()
            
            # Check specific search endpoint
            test_response = requests.get(
                f"{self.instance_url}/search?q=test", 
                headers=self._get_headers(), 
                timeout=5
            )
            test_response.raise_for_status()
            
            logger.info("SearXNG instance is fully accessible")
            return True
        
        except requests.RequestException as e:
            logger.error(f"Connection diagnosis failed: {e}")
            return False

def main():
    client = SearXNGClient()
    
    # Connection Diagnosis
    if not client.diagnose_connection():
        logger.error("SearXNG instance not accessible. Check configuration.")
        return
    
    # Perform search
    results = client.search("Ethereum latest news", num_results=3)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.get('title', 'No Title')} - {result.get('url', 'No URL')}")

if __name__ == "__main__":
    main()