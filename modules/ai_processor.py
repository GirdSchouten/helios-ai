import os
import logging
import torch
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline
)
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class AIProcessor:
    def __init__(self, debug: bool = False, cache_dir: str = './ai_cache'):
        """
        Advanced AI Processor with comprehensive research capabilities
        
        Args:
            debug (bool): Enable verbose logging
            cache_dir (str): Directory for caching model results
        """
        self.debug = debug
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Dynamic model selection
        self.model_name = self._select_ai_model()
        logger.info(f"Using model {self.model_name}")
        
        try:
            self.model, self.tokenizer = self._load_model()
            logger.info("Model init success")
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            self.model = None
            self.tokenizer = None
    
    @staticmethod
    def _select_ai_model() -> str:
        """
        Intelligent model selection based on system resources
        
        Returns:
            str: Selected model name/path
        """
        ram = psutil.virtual_memory().total / 1e9  # Convert bytes to GB
        
        if ram >= 32:
            return "microsoft/phi-2"  # Most powerful option
        elif ram >= 16:
            return "google/flan-t5-large"
        else:
            return "google/flan-t5-small"
    
    def _load_model(self) -> tuple:
        """
        Robust model loading with dynamic model type detection
        
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            # Dynamically import appropriate model classes
            from transformers import (
                AutoModelForCausalLM,  # For Phi models
                AutoModelForSeq2SeqLM,  # For T5 models
                AutoTokenizer
            )
            
            # Determine model type based on selected model
            model_name = self.model_name
            
            # Select appropriate model class
            if "phi" in model_name.lower():
                model_class = AutoModelForCausalLM
            elif "t5" in model_name.lower():
                model_class = AutoModelForSeq2SeqLM
            else:
                # Fallback to causal LM if unsure
                model_class = AutoModelForCausalLM
            
            # Load model with appropriate configuration
            model = model_class.from_pretrained(
                model_name, 
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            
            # Explicitly set padding token
            if tokenizer.pad_token is None:
                # Priority-based pad token selection
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                elif tokenizer.bos_token is not None:
                    tokenizer.pad_token = tokenizer.bos_token
                else:
                    # Add a new pad token if no existing token works
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    model.resize_token_embeddings(len(tokenizer))
            
            return model, tokenizer
        
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise

    def _try_advanced_load(self, model_name, **kwargs):
        """Advanced model loading with additional configurations"""
        try:
            # Try with Accelerate if available
            from accelerate import dispatch_model
            
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # Attempt to use Accelerate if installed
            if torch.cuda.is_available():
                model = dispatch_model(model)
            
            return model, tokenizer
        except ImportError:
            # Fallback to standard loading if Accelerate is not installed
            return self._try_basic_load(model_name, **kwargs)
    
    def _try_basic_load(self, model_name, **kwargs):
        """Basic model loading with minimal requirements"""
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    
    def analyze_content(self, title: str, content: str) -> Dict[str, Any]:
        """
        Comprehensive content analysis with multiple research dimensions
        
        Args:
            title (str): Article title
            content (str): Article content
        
        Returns:
            Multi-dimensional AI insights
        """
        if not self.model or not self.tokenizer:
            return self._default_analysis()
        
        try:
            # Comprehensive research prompt
            prompt = f"""
            Perform a comprehensive analysis of the following content:
            
            Title: {title}
            Content: {content}
            
            Provide insights on:
            1. SENTIMENT: Overall emotional tone (Positive/Negative/Neutral)
            2. MARKET_SIGNALS: Current market trend (Bullish/Bearish/Neutral)
            3. KEY_NEWS: Most significant news points
            4. RISK_ASSESSMENT: Potential market impact
            
            Format your response in clear, concise sections.
            """
            
            # Generate comprehensive analysis
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
            outputs = self.model.generate(
                **inputs, 
                max_length=300,  # Increased to capture more details
                num_return_sequences=1,
                temperature=0.3,  # Balance between creativity and focus
                do_sample=True
            )
            
            # Decode and parse the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse the AI-generated response
            return self._parse_comprehensive_analysis(response)
        
        except Exception as e:
            logger.error(f"Comprehensive analysis error: {e}")
            return self._default_analysis()
    
    def _parse_comprehensive_analysis(self, response: str) -> Dict[str, Any]:
        """
        Parse the AI-generated comprehensive analysis
        
        Args:
            response (str): AI-generated analysis text
        
        Returns:
            Structured analysis dictionary
        """
        analysis = {
            'sentiment': self._extract_sentiment(response),
            'market_signals': self._extract_market_signals(response),
            'key_news': self._extract_key_news(response),
            'risk_assessment': self._extract_risk_assessment(response)
        }
        
        return analysis
    
    def _extract_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Extract sentiment from analysis text
        
        Args:
            text (str): Analysis text
        
        Returns:
            Sentiment analysis dictionary
        """
        sentiment_map = {
            'positive': {'label': 'POSITIVE', 'score': 0.8},
            'negative': {'label': 'NEGATIVE', 'score': 0.8},
            'neutral': {'label': 'NEUTRAL', 'score': 0.5}
        }
        
        # Use simple keyword matching
        if 'positive' in text.lower():
            return sentiment_map['positive']
        elif 'negative' in text.lower():
            return sentiment_map['negative']
        
        return sentiment_map['neutral']
    
    def _extract_market_signals(self, text: str) -> Dict[str, str]:
        """
        Extract market signals from analysis text
        
        Args:
            text (str): Analysis text
        
        Returns:
            Market signals dictionary
        """
        text_lower = text.lower()
        
        if 'bullish' in text_lower:
            return {
                'trend': 'bullish',
                'volatility': 'medium'
            }
        elif 'bearish' in text_lower:
            return {
                'trend': 'bearish',
                'volatility': 'high'
            }
        
        return {
            'trend': 'neutral',
            'volatility': 'low'
        }
    
    def _extract_key_news(self, text: str, max_items: int = 3) -> List[str]:
        """
        Extract key news points from analysis text
        
        Args:
            text (str): Analysis text
            max_items (int): Maximum number of news items to extract
        
        Returns:
            List of key news points
        """
        # Simple approach to extract key news
        sentences = text.split('.')
        key_news = [
            sentence.strip() 
            for sentence in sentences 
            if len(sentence.strip()) > 30  # Avoid very short sentences
        ]
        
        return key_news[:max_items]
    
    def _extract_risk_assessment(self, text: str) -> Dict[str, str]:
        """
        Extract risk assessment from analysis text
        
        Args:
            text (str): Analysis text
        
        Returns:
            Risk assessment dictionary
        """
        text_lower = text.lower()
        
        if 'high risk' in text_lower:
            return {
                'level': 'high',
                'recommendation': 'Caution advised'
            }
        elif 'moderate risk' in text_lower:
            return {
                'level': 'moderate',
                'recommendation': 'Proceed with careful evaluation'
            }
        
        return {
            'level': 'low',
            'recommendation': 'Relatively stable'
        }
    
    def extract_price(self, asset: str, title: str, content: str) -> Optional[float]:
        """
        Intelligently extract price using dynamically selected model
        
        Args:
            title (str): Article title
            content (str): Article content
        
        Returns:
            Extracted price or None
        """
        if not self.model or not self.tokenizer:
            return None
        
        # Combine title and content for comprehensive context
        full_text = f"Title: {title}\n\nContent: {content}"
        
        try:
            # Prepare comprehensive price extraction prompt
            prompt = f"""
            Extract the most recent and accurate price in USD of the asset {asset} from the following text.
            Ignore if it is not about the price of {asset} crypto-currency

            Guidelines:
            - Look for explicit price mentions in dollars value only
            - Prefer the most recent and credible price
            - Return ONLY the numeric price value
            - If no clear price is found, return 'None'
            
            Text to Analyze:
            {full_text}
            
            Extracted Current Price (USD):"""
            
            # Tokenization with safe padding
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=1024,  # Increased max length
                truncation=True,
                padding=True,
                add_special_tokens=True
            )
            
            # Ensure inputs are on the correct device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Dynamic generation parameters based on model type
            generation_params = {
                'max_new_tokens': 50,  # Limit new tokens generated
                'num_return_sequences': 1,
                'do_sample': True,
                'temperature': 0.2,  # More focused generation
                'top_k': 50,
                'top_p': 0.95,
            }
            
            # Select appropriate generation method based on model type
            if "t5" in self.model_name.lower():
                # T5-specific generation
                outputs = self.model.generate(**inputs, **generation_params)
            else:
                # Causal LM generation (like Phi)
                outputs = self.model.generate(**inputs, **generation_params)
            
            # Decode the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            print(response)
            
            # Advanced price extraction with multiple parsing strategies
            extracted_prices = []
            
            # Regular expression for finding prices
            import re
            price_patterns = [
                r'\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)',  # Handles \$1,234.56
                r'(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?) USD',   # Handles 1,234.56 USD
                r'price\s*[is]*\s*\$?\s?(\d+(?:,\d{3})*(?:\.\d{1,2})?)'  # Handles "price is \$1,234.56"
            ]
            
            for pattern in price_patterns:
                matches = re.findall(pattern, response, re.IGNORECASE)
                if matches:
                    for match in matches:
                        # Clean and convert to float
                        try:
                            cleaned_price = match.replace(',', '').strip()
                            price = float(cleaned_price)
                            
                            # Sanity check for reasonable price range
                            if 0 < price < 100000:
                                extracted_prices.append(price)
                        except (ValueError, TypeError):
                            pass
            
            # Return the first valid price, or None
            if extracted_prices:
                return extracted_prices[0]
            
            # Fallback parsing of the raw response
            try:
                # Try direct numeric conversion
                cleaned_response = re.sub(r'[^\d.]', '', response)
                fallback_price = float(cleaned_response)
                
                if 0 < fallback_price < 100000:
                    return fallback_price
            except (ValueError, TypeError):
                pass
            
            return None
        
        except Exception as e:
            logger.warning(f"Price extraction error: {e}")
            return None
    
    def _default_analysis(self) -> Dict[str, Any]:
        """
        Provide default analysis when processing fails
        
        Returns:
            Default analysis dictionary
        """
        return {
            'sentiment': {'label': 'NEUTRAL', 'score': 0.5},
            'market_signals': {
                'trend': 'neutral',
                'volatility': 'low'
            },
            'key_news': [],
            'risk_assessment': {
                'level': 'low',
                'recommendation': 'No significant concerns'
            }
        }

def main():
    # Demonstration of comprehensive analysis
    processor = AIProcessor(debug=True)
    
    # Sample content for testing
    sample_contents = [
        {
            "title": "Ethereum Market Update",
            "content": "Ethereum is showing strong growth potential with recent blockchain innovations. The current price stands around \$3,456.78, indicating positive market sentiment."
        },
        {
            "title": "Crypto Market Volatility",
            "content": "Recent regulatory changes and market fluctuations suggest a cautious approach to cryptocurrency investments."
        }
    ]
    
    for content in sample_contents:
        print(f"\nTitle: {content['title']}")
        
        # Price extraction
        price = processor.extract_price(content['title'], content['content'])
        print(f"Extracted Price: ${price}" if price else "No price found")
        
        # Comprehensive analysis
        analysis = processor.analyze_content(content['title'], content['content'])
        print("\nComprehensive Analysis:")
        print(json.dumps(analysis, indent=2))

if __name__ == "__main__":
    main()