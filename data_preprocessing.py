import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def clean_text(text):
    """
    Clean the text by removing special characters, links, and emojis.
    
    Parameters:
    text (str): Text to clean
    
    Returns:
    str: Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove emojis and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def remove_stopwords(text):
    """
    Remove stopwords from text.
    
    Parameters:
    text (str): Text to process
    
    Returns:
    str: Text with stopwords removed
    """
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)

def lemmatize_text(text):
    """
    Lemmatize text to reduce words to their base form.
    
    Parameters:
    text (str): Text to lemmatize
    
    Returns:
    str: Lemmatized text
    """
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized_tokens)

def preprocess_comments(comments):
    """
    Preprocess a list of comments for sentiment analysis.
    
    Parameters:
    comments (list): List of comment texts
    
    Returns:
    list: List of preprocessed comments
    """
    preprocessed_comments = []
    
    for comment in comments:
        if comment and isinstance(comment, str):
            # Apply preprocessing steps
            cleaned_comment = clean_text(comment)
            no_stopwords = remove_stopwords(cleaned_comment)
            lemmatized = lemmatize_text(no_stopwords)
            
            # Add to processed list
            preprocessed_comments.append(lemmatized)
        else:
            # Handle empty or non-string comments
            preprocessed_comments.append("")
    
    return preprocessed_comments
