import re
import os
import time
import pandas as pd
from datetime import datetime

def validate_youtube_url(url):
    """
    Validate if the URL is a valid YouTube URL.
    
    Parameters:
    url (str): URL to validate
    
    Returns:
    bool: True if valid, False otherwise
    """
    # Regular expression to match YouTube URLs
    youtube_regex = (
        r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
        r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    )
    
    match = re.match(youtube_regex, url)
    return match is not None

def format_timestamp(timestamp):
    """
    Format a timestamp in a human-readable format.
    
    Parameters:
    timestamp (datetime): Timestamp to format
    
    Returns:
    str: Formatted timestamp
    """
    return timestamp.strftime('%Y-%m-%d %H:%M:%S')

def truncate_text(text, max_length=100):
    """
    Truncate text to a maximum length.
    
    Parameters:
    text (str): Text to truncate
    max_length (int): Maximum length
    
    Returns:
    str: Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - 3] + '...'

def get_sentiment_color(sentiment):
    """
    Get color for a sentiment.
    
    Parameters:
    sentiment (str): Sentiment label
    
    Returns:
    str: Color hex code
    """
    color_map = {
        'positive': '#4CAF50',  # Green
        'neutral': '#FFC107',   # Amber
        'negative': '#F44336'   # Red
    }
    
    return color_map.get(sentiment, '#7b2cbf')  # Default to purple if not found

def save_results(comments_df, video_id):
    """
    Save analysis results to a CSV file.
    
    Parameters:
    comments_df (DataFrame): DataFrame with analyzed comments
    video_id (str): YouTube video ID
    
    Returns:
    str: Path to the saved file
    """
    # Create results directory if it doesn't exist
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{results_dir}/analysis_{video_id}_{timestamp}.csv"
    
    # Save to CSV
    comments_df.to_csv(filename, index=False)
    
    return filename

def calculate_sentiment_statistics(comments_df):
    """
    Calculate sentiment statistics from a DataFrame.
    
    Parameters:
    comments_df (DataFrame): DataFrame with analyzed comments
    
    Returns:
    dict: Dictionary with sentiment statistics
    """
    # Get total counts
    total_comments = len(comments_df)
    
    # Get sentiment counts
    sentiment_counts = comments_df['Sentiment'].value_counts().to_dict()
    
    # Calculate percentages
    sentiment_percentages = {
        sentiment: count / total_comments * 100
        for sentiment, count in sentiment_counts.items()
    }
    
    # Get average sentiment score
    avg_sentiment_score = comments_df['Sentiment_Score'].mean()
    
    # Return statistics
    return {
        'total_comments': total_comments,
        'sentiment_counts': sentiment_counts,
        'sentiment_percentages': sentiment_percentages,
        'avg_sentiment_score': avg_sentiment_score
    }

def rate_limit_api_calls(api_calls, max_calls=10, time_period=60):
    """
    Rate limit API calls.
    
    Parameters:
    api_calls (list): List of timestamps of API calls
    max_calls (int): Maximum number of calls in the time period
    time_period (int): Time period in seconds
    
    Returns:
    bool: True if the call is allowed, False otherwise
    """
    # Get current time
    current_time = time.time()
    
    # Remove old calls
    api_calls = [call for call in api_calls if current_time - call < time_period]
    
    # Check if maximum calls reached
    if len(api_calls) >= max_calls:
        return False
    
    # Add current call
    api_calls.append(current_time)
    
    return True
