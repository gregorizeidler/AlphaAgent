"""
Sentiment Analysis using FinBERT and GPT
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Optional
import logging
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinBERTSentimentAnalyzer:
    """
    Sentiment analysis using FinBERT model fine-tuned on financial texts
    """
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Args:
            model_name: HuggingFace model name for FinBERT
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded FinBERT model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading FinBERT model: {e}")
            raise
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment scores (positive, negative, neutral)
        """
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # FinBERT outputs: [positive, negative, neutral]
            scores = predictions[0].cpu().numpy()
            
            return {
                'positive': float(scores[0]),
                'negative': float(scores[1]),
                'neutral': float(scores[2]),
                'compound': float(scores[0] - scores[1])  # Overall sentiment
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0}
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment of multiple texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment dictionaries
        """
        return [self.analyze_text(text) for text in texts]
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get sentence embedding from FinBERT
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector (768-dimensional)
        """
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.base_model(**inputs)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return np.zeros(768)


class GPTSentimentAnalyzer:
    """
    Advanced sentiment analysis using GPT models
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Args:
            api_key: OpenAI API key (if None, loads from environment)
            model: GPT model to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        
        if not self.api_key:
            logger.warning("No OpenAI API key found. GPT analysis will be disabled.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"Initialized GPT analyzer with model: {model}")
    
    def analyze_news(self, news_items: List[Dict[str, str]], ticker: str) -> Dict[str, any]:
        """
        Analyze news items using GPT to extract market insights
        
        Args:
            news_items: List of news dictionaries with 'title' key
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with sentiment analysis and insights
        """
        if not self.client or not news_items:
            return {
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'key_topics': [],
                'risk_factors': [],
                'summary': 'No analysis available'
            }
        
        try:
            # Prepare news context
            news_text = "\n".join([
                f"- {item['title']}" for item in news_items[:10]
            ])
            
            prompt = f"""Analyze the following news headlines about {ticker} stock and provide:
1. Overall sentiment score (-1 to 1, where -1 is very negative, 0 is neutral, 1 is very positive)
2. Confidence level (0 to 1)
3. Key topics mentioned (max 5)
4. Potential risk factors (max 3)
5. Brief summary (max 50 words)

News Headlines:
{news_text}

Respond in JSON format:
{{
    "sentiment_score": <float>,
    "confidence": <float>,
    "key_topics": [<string>, ...],
    "risk_factors": [<string>, ...],
    "summary": "<string>"
}}
"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial analyst expert in market sentiment analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            logger.info(f"GPT sentiment analysis completed for {ticker}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in GPT analysis: {e}")
            return {
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'key_topics': [],
                'risk_factors': [],
                'summary': f'Analysis error: {str(e)}'
            }
    
    def analyze_market_context(self, ticker: str, price_data: Dict, fundamentals: Dict) -> str:
        """
        Use GPT to provide market context and trading insights
        
        Args:
            ticker: Stock ticker
            price_data: Recent price movements
            fundamentals: Fundamental metrics
            
        Returns:
            Text analysis with insights
        """
        if not self.client:
            return "GPT analysis not available"
        
        try:
            prompt = f"""As a trading analyst, provide a brief analysis of {ticker}:

Recent Price Data:
- Current Price: ${price_data.get('current_price', 'N/A')}
- 5-day change: {price_data.get('change_5d', 'N/A')}%
- 20-day change: {price_data.get('change_20d', 'N/A')}%

Fundamentals:
- P/E Ratio: {fundamentals.get('pe_ratio', 'N/A')}
- Debt/Equity: {fundamentals.get('debt_to_equity', 'N/A')}
- ROE: {fundamentals.get('roe', 'N/A')}

Provide a 2-3 sentence trading context."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a quantitative trading analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=150
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in market context analysis: {e}")
            return "Market context analysis unavailable"


class MultiModalSentimentAnalyzer:
    """
    Combines FinBERT and GPT for comprehensive sentiment analysis
    """
    
    def __init__(self):
        """Initialize both FinBERT and GPT analyzers"""
        self.finbert = FinBERTSentimentAnalyzer()
        self.gpt = GPTSentimentAnalyzer()
    
    def analyze_comprehensive(
        self, 
        news_items: List[Dict[str, str]], 
        ticker: str
    ) -> Dict[str, any]:
        """
        Comprehensive sentiment analysis using both models
        
        Args:
            news_items: List of news items
            ticker: Stock ticker
            
        Returns:
            Combined sentiment analysis
        """
        # FinBERT analysis on individual news items
        finbert_sentiments = []
        embeddings = []
        
        for item in news_items[:10]:  # Analyze top 10 news items
            text = item.get('title', '')
            if text:
                sentiment = self.finbert.analyze_text(text)
                embedding = self.finbert.get_embedding(text)
                finbert_sentiments.append(sentiment)
                embeddings.append(embedding)
        
        # Average FinBERT sentiments
        avg_finbert = {
            'positive': np.mean([s['positive'] for s in finbert_sentiments]) if finbert_sentiments else 0.0,
            'negative': np.mean([s['negative'] for s in finbert_sentiments]) if finbert_sentiments else 0.0,
            'neutral': np.mean([s['neutral'] for s in finbert_sentiments]) if finbert_sentiments else 1.0,
            'compound': np.mean([s['compound'] for s in finbert_sentiments]) if finbert_sentiments else 0.0,
        }
        
        # Average embeddings
        avg_embedding = np.mean(embeddings, axis=0) if embeddings else np.zeros(768)
        
        # GPT analysis
        gpt_analysis = self.gpt.analyze_news(news_items, ticker)
        
        # Combine results
        combined_sentiment = (avg_finbert['compound'] + gpt_analysis.get('sentiment_score', 0.0)) / 2
        
        return {
            'finbert': avg_finbert,
            'gpt': gpt_analysis,
            'combined_sentiment': float(combined_sentiment),
            'embedding': avg_embedding,
            'news_count': len(news_items)
        }


if __name__ == "__main__":
    # Test sentiment analyzers
    sample_news = [
        {'title': 'Apple reports record quarterly earnings, beats expectations'},
        {'title': 'Concerns grow over supply chain disruptions'},
        {'title': 'New product launch receives positive reviews'},
    ]
    
    analyzer = MultiModalSentimentAnalyzer()
    result = analyzer.analyze_comprehensive(sample_news, "AAPL")
    
    print("Sentiment Analysis Results:")
    print(f"FinBERT Compound: {result['finbert']['compound']:.3f}")
    print(f"GPT Sentiment: {result['gpt']['sentiment_score']:.3f}")
    print(f"Combined Sentiment: {result['combined_sentiment']:.3f}")
    print(f"Embedding Shape: {result['embedding'].shape}")

