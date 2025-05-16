import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources - ensure all required resources are available
for resource in ['punkt', 'stopwords', 'wordnet']:
    try:
        nltk.download(resource, quiet=True)
    except Exception as e:
        print(f"Error downloading NLTK resource {resource}: {e}")

# Define a simple tokenizer function for fallback
def simple_tokenize(text):
    """Split text on whitespace and punctuation"""
    if not isinstance(text, str):
        return []
    # Basic cleaning
    text = text.lower()
    # Split on whitespace and remove empty strings
    return [token for token in re.split(r'[^\w]', text) if token]

# Try to use NLTK's tokenizer, but don't rely on it
def safe_word_tokenize(text):
    """Safely tokenize text with fallback to simple tokenizer"""
    if not isinstance(text, str):
        return []
    try:
        return word_tokenize(text)
    except Exception as e:
        print(f"Falling back to simple tokenizer: {e}")
        return simple_tokenize(text)

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
    try:
        # Try to get stopwords, with fallback
        try:
            stop_words = set(stopwords.words('english'))
        except:
            # Common English stopwords as fallback
            stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
                         'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 
                         'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 
                         'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 
                         'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 
                         'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
                         'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 
                         'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
                         'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
                         'through', 'during', 'before', 'after', 'above', 'below', 'to', 
                         'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 
                         'again', 'further', 'then', 'once', 'here', 'there', 'when', 
                         'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
                         'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 
                         'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 
                         'just', 'don', 'should', 'now'}

        # Use safe tokenization to handle errors
        tokens = safe_word_tokenize(text)
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        return ' '.join(filtered_tokens)
    except Exception as e:
        print(f"Error removing stopwords: {e}")
        return text

def lemmatize_text(text):
    """
    Lemmatize text to reduce words to their base form.
    
    Parameters:
    text (str): Text to lemmatize
    
    Returns:
    str: Lemmatized text
    """
    try:
        # Use safe tokenization
        tokens = safe_word_tokenize(text)
        
        # Try to lemmatize with WordNetLemmatizer
        try:
            lemmatizer = WordNetLemmatizer()
            lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
        except Exception as e:
            print(f"Lemmatization failed, using original tokens: {e}")
            lemmatized_tokens = tokens
            
        return ' '.join(lemmatized_tokens)
    except Exception as e:
        print(f"Error lemmatizing text: {e}")
        return text

def preprocess_comments(comments):
    """
    Preprocess a list of comments for sentiment analysis.
    
    Parameters:
    comments (list): List of comment texts
    
    Returns:
    list: List of preprocessed comments
    """
    preprocessed_comments = []
    
    try:
        for comment in comments:
            try:
                if comment and isinstance(comment, str):
                    # Apply preprocessing steps with error handling
                    try:
                        cleaned_comment = clean_text(comment)
                    except Exception as e:
                        print(f"Error cleaning text: {e}")
                        cleaned_comment = comment
                        
                    try:
                        no_stopwords = remove_stopwords(cleaned_comment)
                    except Exception as e:
                        print(f"Error removing stopwords: {e}")
                        no_stopwords = cleaned_comment
                        
                    try:
                        lemmatized = lemmatize_text(no_stopwords)
                    except Exception as e:
                        print(f"Error lemmatizing text: {e}")
                        lemmatized = no_stopwords
                    
                    # Add to processed list
                    preprocessed_comments.append(lemmatized)
                else:
                    # Handle empty or non-string comments
                    preprocessed_comments.append("")
            except Exception as e:
                print(f"Error preprocessing comment: {e}")
                preprocessed_comments.append("")
    except Exception as e:
        print(f"Error in preprocess_comments: {e}")
        # Return original comments if preprocessing fails completely
        return [str(comment) if comment is not None else "" for comment in comments]
    
    return preprocessed_comments
