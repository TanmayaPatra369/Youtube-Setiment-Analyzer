import re
import pandas as pd
from collections import Counter

def extract_ngrams(text_list, n=2, min_count=2, max_phrases=15):
    """
    Extract common phrases (n-grams) from a list of texts

    Parameters:
    text_list (list): List of text strings
    n (int): Size of n-grams to extract (2 for bigrams, 3 for trigrams)
    min_count (int): Minimum count to include a phrase
    max_phrases (int): Maximum number of phrases to return

    Returns:
    list: List of tuples (phrase, count) sorted by frequency
    """
    if not text_list or len(text_list) == 0:
        return []

    # Clean texts and convert to lowercase
    cleaned_texts = []
    for text in text_list:
        if isinstance(text, str):
            # Remove special characters except spaces
            text = re.sub(r'[^\w\s]', '', text.lower())
            cleaned_texts.append(text)

    # Extract n-grams
    all_ngrams = []

    for text in cleaned_texts:
        words = text.split()
        if len(words) >= n:
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                all_ngrams.append(ngram)

    # Count frequencies
    ngram_counter = Counter(all_ngrams)

    # Filter by minimum count
    filtered_ngrams = [(phrase, count) for phrase, count in ngram_counter.items() 
                       if count >= min_count and len(phrase.split()) == n]

    # Sort by frequency and limit to max_phrases
    top_ngrams = sorted(filtered_ngrams, key=lambda x: x[1], reverse=True)[:max_phrases]

    return top_ngrams

def get_common_phrases(comments, sentiment=None):
    """
    Get common phrases for given comments, optionally filtered by sentiment

    Parameters:
    comments (DataFrame): DataFrame containing comments and sentiments
    sentiment (str, optional): Filter comments by this sentiment

    Returns:
    dict: Dictionary with bigrams and trigrams
    """
    if sentiment:
        filtered_comments = comments[comments['Sentiment'] == sentiment]['Comment'].tolist()
    else:
        filtered_comments = comments['Comment'].tolist()

    # Extract bigrams and trigrams
    bigrams = extract_ngrams(filtered_comments, n=2, min_count=2)
    trigrams = extract_ngrams(filtered_comments, n=3, min_count=2)

    return {
        'bigrams': bigrams,
        'trigrams': trigrams
    }

def format_phrases_html(phrases, title, color):
    """
    Format phrases for HTML display

    Parameters:
    phrases (list): List of (phrase, count) tuples
    title (str): Title for the section
    color (str): CSS color for the phrases

    Returns:
    str: HTML string for displaying phrases
    """
    if not phrases:
        return f"<h4>{title}</h4><p>No common phrases found.</p>"

    html = f"<h4>{title}</h4>"
    html += "<div style='max-height: 200px; overflow-y: auto;'>"

    for phrase, count in phrases:
        # Calculate font size based on count (between 90% and 130%)
        max_count = phrases[0][1] if phrases else 1
        size_pct = 90 + int(40 * (count / max_count))

        html += f"<div style='margin: 5px 0; font-size: {size_pct}%; color: {color};'>"
        html += f"<span style='font-weight: bold;'>{phrase}</span> <span style='color: gray;'>({count})</span>"
        html += "</div>"

    html += "</div>"
    return html
