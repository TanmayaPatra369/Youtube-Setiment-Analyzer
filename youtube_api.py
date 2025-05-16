import os
import re
import pandas as pd
import googleapiclient.discovery
from googleapiclient.errors import HttpError


def extract_video_id(url):
    """
    Extract the video ID from a YouTube URL.
    
    Parameters:
    url (str): The YouTube URL
    
    Returns:
    str: The video ID or None if not found
    """
    # Regular expressions to match different YouTube URL formats
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]{11})',  # Standard and shortened
        r'youtube\.com\/embed\/([a-zA-Z0-9_-]{11})',  # Embed URL
        r'youtube\.com\/v\/([a-zA-Z0-9_-]{11})'  # Old embed URL
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def get_video_comments(video_id, max_comments=500):
    """
    Fetch comments from a YouTube video using the YouTube Data API.
    
    Parameters:
    video_id (str): The YouTube video ID
    max_comments (int): Maximum number of comments to retrieve (default: 500)
    
    Returns:
    pandas.DataFrame: DataFrame containing the comments
    """
    try:
        # Get API key from environment variable
        api_key = os.getenv("YOUTUBE_API_KEY",
                            "AIzaSyDq0s_FcHnLJbW8skr-nU4BEg4fbsTzmuo")

        if not api_key:
            # Instead of using a mock API, we'll use a sample dataset if API key is not available
            return get_sample_comments()

        # Initialize the YouTube API client
        youtube = googleapiclient.discovery.build("youtube",
                                                  "v3",
                                                  developerKey=api_key,
                                                  cache_discovery=False)

        # Get comments
        comments = []
        next_page_token = None

        while len(comments) < max_comments:
            # Make API request to get comments
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),
                pageToken=next_page_token,
                textFormat="plainText").execute()

            # Extract comments from response
            for item in response["items"]:
                comment_text = item["snippet"]["topLevelComment"]["snippet"][
                    "textDisplay"]
                comments.append(comment_text)

            # Check if there are more pages
            next_page_token = response.get("nextPageToken")
            if not next_page_token or len(comments) >= max_comments:
                break

        # Create DataFrame
        comments_df = pd.DataFrame({"Comment": comments})
        return comments_df

    except HttpError as e:
        if e.resp.status == 403:
            # API key issues
            print(f"API Error: {e}")
            return get_sample_comments()
        elif e.resp.status == 404:
            # Video not found
            print(f"Video not found: {e}")
            return None
        else:
            # Other errors
            print(f"YouTube API Error: {e}")
            return None
    except Exception as e:
        print(f"Error retrieving comments: {e}")
        return None


def get_sample_comments():
    """
    Return a sample of comments from the dataset if the API key is not available.
    
    Returns:
    pandas.DataFrame: DataFrame containing sample comments
    """
    try:
        # Load comments from CSV file
        comments_df = pd.read_csv('attached_assets/YoutubeCommentsDataSet.csv')
        return comments_df[['Comment']].sample(min(500, len(comments_df)))
    except Exception as e:
        print(f"Error loading sample comments: {e}")
        # Create a small dataframe with a few comments as fallback
        fallback_comments = [
            "This video was really helpful, thank you!",
            "I didn't like this content at all.",
            "Interesting perspective, I'm not sure if I agree.",
            "Great production quality as always!",
            "This is the worst video I've seen on this topic."
        ]
        return pd.DataFrame({"Comment": fallback_comments})
