# main.py
import logging
from typing import List, Dict, Optional
from modules.searxng_client import SearXNGClient
from modules.data_parser import fetch_and_parse_article
from modules.ai_processor import AIProcessor
import json
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class AssetResearchEngine:
    def __init__(self, cache_dir: str = './research_cache'):
        """
        Initialize research components with caching mechanism
        
        Args:
            cache_dir (str): Directory to store research caches
        """
        self.searxng_client = SearXNGClient()
        self.ai_processor = AIProcessor()
        self.cache_dir = cache_dir
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
    
    def research_asset(
        self, 
        asset: str, 
        research_types: List[str] = ['price', 'news', 'sentiment', 'market_signals'],
        num_results: int = 5
    ) -> Dict:
        """
        Comprehensive decentralized asset research
        
        Args:
            asset (str): Asset to research
            research_types (List[str]): Types of research to perform
            num_results (int): Number of search results per type
        
        Returns:
            Comprehensive research report
        """
        try: 
            # Perform comprehensive research
            comprehensive_report = {
                'timestamp': datetime.now().isoformat(),
                'asset': asset,
                'research_types': research_types
            }
            
            # Price Research
            if 'price' in research_types:
                comprehensive_report['price_analysis'] = self._aggregate_price_from_sources(asset, num_results)
            
            # News and Sentiment Analysis
            if 'news' in research_types or 'sentiment' in research_types:
                comprehensive_report['news_analysis'] = self._analyze_news_and_sentiment(asset, num_results)
            
            # Market Signals
            if 'market_signals' in research_types:
                comprehensive_report['market_signals'] = self._extract_market_signals(asset, num_results)
            
            # Cache the result
            self._cache_result(asset, comprehensive_report)
            
            return comprehensive_report
        
        except Exception as e:
            logger.error(f"Comprehensive asset research failed: {e}")
            return {}
    
    def _aggregate_price_from_sources(self, asset: str, num_results: int) -> Optional[Dict]:
        """
        Aggregate price from multiple search sources
        
        Args:
            asset (str): Asset to find price for
            num_results (int): Number of search results
        
        Returns:
            Price analysis dictionary
        """
        price_queries = [
            f"current price of {asset} in usd",
            f"{asset} latest trading value in usd",
            f"{asset} price today in usd"
        ]
        
        all_price_results = []
        source_details = []
        
        for query in price_queries:
            search_results = self.searxng_client.search(query, num_results=num_results)
            
            for result in search_results:
                try:
                    # Fetch full article content
                    article_text = fetch_and_parse_article(result['url'])
                    
                    # Extract price using AI processor
                    price = self._extract_price_from_content(
                        asset,
                        title=result['title'], 
                        content=article_text
                    )
                    
                    if price:
                        all_price_results.append(price)
                        source_details.append({
                            'url': result['url'],
                            'title': result['title'],
                            'price': price
                        })
                except Exception as e:
                    logger.warning(f"Price extraction error for {result['url']}: {e}")
        
        # Statistical price aggregation
        if all_price_results:
            return {
                'prices': all_price_results,
                'average_price': sum(all_price_results) / len(all_price_results),
                'price_range': {
                    'min': min(all_price_results),
                    'max': max(all_price_results)
                },
                'sources_count': len(all_price_results),
                'source_details': source_details
            }
        
        return None
    
    def _extract_price_from_content(self, asset: str, title: str, content: str) -> Optional[float]:
        """
        Extract price from article content using AI
        
        Args:
            title (str): Article title
            content (str): Article text
        
        Returns:
            Extracted price or None
        """
        try:
            # Use AI processor to extract price
            price_insight = self.ai_processor.extract_price(asset, title, content)
            return price_insight
        except Exception as e:
            logger.warning(f"Price extraction failed: {e}")
            return None
    
    def _analyze_news_and_sentiment(self, asset: str, num_results: int) -> Dict:
        """
        Comprehensive news and sentiment analysis
        
        Args:
            asset (str): Asset to analyze
            num_results (int): Number of search results
        
        Returns:
            News and sentiment analysis dictionary
        """
        search_results = self.searxng_client.search(
            f"{asset} latest news and market sentiment", 
            num_results=num_results
        )
        
        news_analysis = {
            'articles': [],
            'overall_sentiment': {
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0
            },
            'key_topics': []
        }
        
        for result in search_results:
            # Fetch full article content
            article_text = fetch_and_parse_article(result['url'])
            
            # Analyze article
            article_insight = self.ai_processor.analyze_content(
                title=result['title'],
                content=article_text
            )
            
            # Aggregate insights
            news_analysis['articles'].append({
                'title': result['title'],
                'url': result['url'],
                'sentiment': article_insight.get('sentiment', {}).get('label', 'neutral')
            })
            
            # Update sentiment counts
            sentiment_label = article_insight.get('sentiment', {}).get('label', 'neutral')
            if sentiment_label == 'POSITIVE':
                news_analysis['overall_sentiment']['positive_count'] += 1
            elif sentiment_label == 'NEGATIVE':
                news_analysis['overall_sentiment']['negative_count'] += 1
            else:
                news_analysis['overall_sentiment']['neutral_count'] += 1
            
            # Collect key topics
            news_analysis['key_topics'].extend(article_insight.get('key_topics', []))
        
        # Deduplicate key topics
        news_analysis['key_topics'] = list(set(news_analysis['key_topics']))[:5]
        
        return news_analysis
    
    def _extract_market_signals(self, asset: str, num_results: int) -> Dict:
        """
        Extract broader market signals and indicators
        
        Args:
            asset (str): Asset to analyze
            num_results (int): Number of search results
        
        Returns:
            Market signals dictionary
        """
        market_queries = [
            f"{asset} market trends",
            f"{asset} investment outlook",
            f"{asset} technical analysis"
        ]
        
        market_signals = {
            'trend_indicators': [],
            'volatility_signals': [],
            'cumulative_trend': 'neutral'
        }
        
        trend_counts = {
            'bullish': 0,
            'bearish': 0,
            'neutral': 0
        }
        
        for query in market_queries:
            results = self.searxng_client.search(query, num_results=num_results)
            
            for result in results:
                # Fetch and analyze content
                article_text = fetch_and_parse_article(result['url'])
                
                # Analyze article for market signals
                article_insight = self.ai_processor.analyze_content(
                    title=result['title'],
                    content=article_text
                )
                
                # Extract market signals
                market_signals_detail = article_insight.get('market_signals', {})
                
                # Aggregate signals
                market_signals['trend_indicators'].append(market_signals_detail.get('trend', 'neutral'))
                market_signals['volatility_signals'].append(market_signals_detail.get('volatility', 'low'))
                
                # Count trends
                trend_counts[market_signals_detail.get('trend', 'neutral')] += 1
        
        # Determine cumulative trend
        if trend_counts['bullish'] > trend_counts['bearish'] and trend_counts['bullish'] > trend_counts['neutral']:
            market_signals['cumulative_trend'] = 'bullish'
        elif trend_counts['bearish'] > trend_counts['bullish'] and trend_counts['bearish'] > trend_counts['neutral']:
            market_signals['cumulative_trend'] = 'bearish'
        
        return market_signals
    
    def _check_cache(self, asset: str) -> Optional[Dict]:
        """
        Check if cached research exists
        
        Args:
            asset (str): Asset to check in cache
        
        Returns:
            Cached research or None
        """
        cache_file = os.path.join(self.cache_dir, f"{asset.replace(' ', '_')}_research.json")
        
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                
                # Check cache freshness (e.g., less than 1 hour old)
                cached_time = datetime.fromisoformat(cached_data.get('timestamp', ''))
                if (datetime.now() - cached_time).total_seconds() < 3600:
                    return cached_data
        
        return None
    
    def _cache_result(self, asset: str, result: Dict):
        """
        Cache research result
        
        Args:
            asset (str): Asset researched
            result (Dict): Research results
        """
        cache_file = os.path.join(self.cache_dir, f"{asset.replace(' ', '_')}_research.json")
        
        with open(cache_file, 'w') as f:
            json.dump(result, f, indent=2)

def main():
    # Select assets for research
    assets = [
        "ethereum"
    ]
    
    for asset in assets:
        try:
            # Initialize research engine
            research_engine = AssetResearchEngine()
            
            print(f"\nStarting the analysis of {asset.upper()}")
            # Perform comprehensive research
            # Temporarily focus on price while keeping other research types intact
            research_results = research_engine.research_asset(
                asset, 
                # You can comment/uncomment research types as needed
                research_types=['price'],  # Currently only price
                # research_types=['price', 'news', 'sentiment', 'market_signals'],  # Full research
                num_results=15
            )
            
            # Pretty print results
            print(f"\n🚀 Comprehensive Research Report for {asset.upper()}")
            print(json.dumps(research_results, indent=2))
        
        except Exception as e:
            logger.error(f"Research for {asset} failed: {e}")

if __name__ == "__main__":
    main()