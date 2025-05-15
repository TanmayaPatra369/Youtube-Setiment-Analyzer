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

# Ensure tokenizer is properly loaded
try:
    # Explicitly initialize the tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    # Fallback to simple tokenization if NLTK resources fail
    def simple_tokenize(text):
        return text.split()

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
        stop_words = set(stopwords.words('english'))
        # Use word_tokenize with try/except to handle possible errors
        try:
            tokens = word_tokenize(text)
        except:
            # Use fallback tokenizer if nltk tokenizer fails
            tokens = simple_tokenize(text)
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
        lemmatizer = WordNetLemmatizer()
        # Use word_tokenize with try/except to handle possible errors
        try:
            tokens = word_tokenize(text)
        except:
            # Use fallback tokenizer if nltk tokenizer fails
            tokens = simple_tokenize(text)
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
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
