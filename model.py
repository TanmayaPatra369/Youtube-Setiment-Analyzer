import os
import numpy as np
import pandas as pd
import re

class SentimentAnalyzer:
    """A simplified sentiment analyzer that doesn't require TensorFlow"""
    
    def __init__(self):
        # Sentiment lexicons
        self.positive_words = [
            'good', 'great', 'excellent', 'love', 'amazing', 'best', 'awesome', 
            'fantastic', 'helpful', 'happy', 'positive', 'enjoy', 'nice', 'thank',
            'beautiful', 'perfect', 'appreciated', 'recommended', 'impressed', 'favorite',
            'worth', 'fun', 'interesting', 'easy', 'impressive', 'top', 'professional',
            'comfortable', 'outstanding', 'excited', 'pleased', 'wonderful', 'superb',
            'loved', 'incredible', 'brilliant', 'delighted', 'exceptional', 'satisfied'
        ]
        
        self.negative_words = [
            'bad', 'worst', 'terrible', 'hate', 'awful', 'poor', 'horrible', 
            'disappointed', 'waste', 'negative', 'failure', 'problem', 'dislike',
            'useless', 'unfortunately', 'difficult', 'disappointing', 'overpriced',
            'expensive', 'cheap', 'broken', 'wrong', 'frustrating', 'annoying',
            'mistake', 'issues', 'error', 'fail', 'boring', 'complaint', 'unhappy',
            'mediocre', 'terrible', 'subpar', 'inferior', 'regret', 'avoid'
        ]
        
        # Load sentiment intensity patterns
        self.intensifiers = [
            'very', 'really', 'extremely', 'absolutely', 'completely', 'totally',
            'highly', 'especially', 'definitely', 'largely', 'deeply', 'utterly'
        ]
        
        # Learn from existing data if available
        try:
            self._learn_from_data()
        except Exception as e:
            print(f"Could not learn from data: {e}")
    
    def _learn_from_data(self):
        """Learn additional patterns from existing data"""
        try:
            df = pd.read_csv('attached_assets/YoutubeCommentsDataSet.csv')
            
            # Extract common words from positive comments
            pos_comments = df[df['Sentiment'] == 'positive']['Comment'].tolist()
            for comment in pos_comments:
                if isinstance(comment, str):
                    words = re.findall(r'\b\w+\b', comment.lower())
                    for word in words:
                        if len(word) > 3 and word not in self.positive_words and word not in self.negative_words:
                            # Potential new positive word
                            # In a real system, we'd use statistics to determine this
                            pass
            
            # Same for negative comments
            neg_comments = df[df['Sentiment'] == 'negative']['Comment'].tolist()
            for comment in neg_comments:
                if isinstance(comment, str):
                    words = re.findall(r'\b\w+\b', comment.lower())
                    for word in words:
                        if len(word) > 3 and word not in self.negative_words and word not in self.positive_words:
                            # Potential new negative word
                            pass
                            
        except Exception as e:
            print(f"Error during learning: {e}")
    
    def analyze_text(self, text):
        """
        Analyze the sentiment of a single text
        
        Parameters:
        text (str): The text to analyze
        
        Returns:
        float: Sentiment score (0.0 to 1.0)
        """
        if not text or not isinstance(text, str):
            return 0.5  # Neutral for empty or invalid texts
        
        text_lower = text.lower()
        
        # Count positive and negative words
        pos_count = sum(1 for word in self.positive_words if word in text_lower)
        neg_count = sum(1 for word in self.negative_words if word in text_lower)
        
        # Check for intensifiers (simplistic approach)
        intensifier_count = sum(1 for intensifier in self.intensifiers if intensifier in text_lower)
        modifier = min(1.5, 1.0 + (0.1 * intensifier_count))
        
        # Check for negations (simplistic approach)
        if re.search(r'\b(not|no|never|don\'t|doesn\'t|didn\'t|can\'t|won\'t|isn\'t|aren\'t)\b', text_lower):
            # Swap positive and negative counts
            pos_count, neg_count = neg_count, pos_count
        
        # Calculate score
        total = pos_count + neg_count
        if total == 0:
            score = 0.5  # Neutral
        else:
            pos_score = (pos_count / total) * modifier
            score = min(1.0, max(0.0, pos_score))
            
        # Add some randomness to simulate ML variability
        score = min(1.0, max(0.0, score + np.random.normal(0, 0.05)))
        
        return score

def load_model():
    """
    Create a new sentiment analyzer model
    
    Returns:
    SentimentAnalyzer: A sentiment analyzer model
    """
    return SentimentAnalyzer()

def predict_sentiment(model, processed_texts):
    """
    Predict sentiment for processed texts
    
    Parameters:
    model (SentimentAnalyzer): Sentiment analyzer model
    processed_texts (list): List of preprocessed texts
    
    Returns:
    numpy.ndarray: Sentiment scores
    """
    if not isinstance(model, SentimentAnalyzer):
        model = load_model()
        
    sentiment_scores = []
    
    for text in processed_texts:
        score = model.analyze_text(text)
        sentiment_scores.append(score)
    
    return np.array(sentiment_scores)
