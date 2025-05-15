import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import re
import time
from PIL import Image
from io import BytesIO
import requests

from youtube_api import get_video_comments, extract_video_id
from model import load_model, predict_sentiment
from data_preprocessing import preprocess_comments
from visualizations import create_sentiment_pie_chart, create_sentiment_bar_chart, create_wordcloud

# Set page configuration
st.set_page_config(
    page_title="YouTube Comment Sentiment Analysis",
    page_icon="📊",
    layout="wide",
)

# Add header with title and description
st.title("YouTube Comment Sentiment Analysis")
st.markdown("""
This application uses a Long Short-Term Memory (LSTM) neural network to analyze 
the sentiment of YouTube video comments. Enter a YouTube video URL to get started!
""")

# Sidebar for app navigation and options
with st.sidebar:
    st.header("About")
    st.info("""
    This application performs sentiment analysis on YouTube comments using an LSTM model.
    
    It classifies comments into three categories:
    - 😊 Positive
    - 😐 Neutral
    - 😔 Negative
    
    You'll see a breakdown of sentiment distribution and the actual comments categorized by sentiment.
    """)
    
    st.header("How it works")
    st.markdown("""
    1. Enter a YouTube URL
    2. Our app fetches the comments
    3. The LSTM model analyzes each comment
    4. Results are displayed with visualizations
    """)
    
    # Add images
    try:
        # ML concept illustration
        response = requests.get("https://pixabay.com/get/gbcc37d869de238846409d6c04f59d7c849852171ddf02b2791276bfbd3a777adc069e85c73c18ca58cd9bd525f4cfc5d2946ac40cca71eae5d5a0b73386dbc26_1280.jpg")
        img = Image.open(BytesIO(response.content))
        st.image(img, caption="Machine Learning for Sentiment Analysis", use_column_width=True)
    except Exception as e:
        st.warning("Could not load image")

# Main container for the application
main_container = st.container()

with main_container:
    # Input section
    st.header("Enter YouTube Video URL")
    
    # Input field for YouTube URL
    youtube_url = st.text_input(
        "Paste a YouTube video URL",
        placeholder="https://www.youtube.com/watch?v=..."
    )
    
    # Button to trigger analysis
    analyze_button = st.button("Analyze Comments", type="primary")
    
    # Process when button is clicked
    if analyze_button and youtube_url:
        # Validate the YouTube URL
        video_id = extract_video_id(youtube_url)
        
        if not video_id:
            st.error("Invalid YouTube URL. Please enter a valid YouTube video URL.")
        else:
            # Display progress
            with st.spinner("Fetching comments from YouTube..."):
                try:
                    # Fetch comments
                    comments_df = get_video_comments(video_id)
                    
                    if comments_df is None or comments_df.empty:
                        st.error("Could not retrieve comments for this video. The video might have comments disabled or the API rate limit has been reached.")
                    else:
                        # Display success message
                        st.success(f"Successfully retrieved {len(comments_df)} comments!")
                        
                        # Preprocess comments
                        with st.spinner("Preprocessing comments..."):
                            X_processed = preprocess_comments(comments_df["Comment"].tolist())
                        
                        # Load model and predict
                        with st.spinner("Analyzing sentiment with LSTM model..."):
                            model = load_model()
                            comments_df["Sentiment_Score"] = predict_sentiment(model, X_processed)
                            
                            # Map numerical predictions to sentiment labels
                            def map_sentiment(score):
                                if score < 0.33:
                                    return "negative"
                                elif score < 0.66:
                                    return "neutral"
                                else:
                                    return "positive"
                            
                            comments_df["Sentiment"] = comments_df["Sentiment_Score"].apply(map_sentiment)
                        
                        # Display results
                        st.header("Analysis Results")
                        
                        # Create tabs for different views
                        tab1, tab2, tab3 = st.tabs(["Overview", "Comments Analysis", "Word Analysis"])
                        
                        with tab1:
                            # Overview statistics
                            st.subheader("Sentiment Distribution")
                            
                            # Calculate sentiment distribution
                            sentiment_counts = comments_df["Sentiment"].value_counts().reset_index()
                            sentiment_counts.columns = ["Sentiment", "Count"]
                            
                            # Create columns for charts
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Pie chart
                                fig_pie = create_sentiment_pie_chart(sentiment_counts)
                                st.plotly_chart(fig_pie, use_container_width=True)
                            
                            with col2:
                                # Bar chart
                                fig_bar = create_sentiment_bar_chart(sentiment_counts)
                                st.plotly_chart(fig_bar, use_container_width=True)
                            
                            # Summary statistics
                            st.subheader("Summary Statistics")
                            total_comments = len(comments_df)
                            positive_count = len(comments_df[comments_df["Sentiment"] == "positive"])
                            neutral_count = len(comments_df[comments_df["Sentiment"] == "neutral"])
                            negative_count = len(comments_df[comments_df["Sentiment"] == "negative"])
                            
                            # Display metrics
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Positive Comments", positive_count, f"{positive_count/total_comments:.1%}")
                            col2.metric("Neutral Comments", neutral_count, f"{neutral_count/total_comments:.1%}")
                            col3.metric("Negative Comments", negative_count, f"{negative_count/total_comments:.1%}")
                            
                        with tab2:
                            # Comments Analysis
                            st.subheader("Comments by Sentiment")
                            
                            # Create expanders for each sentiment category
                            with st.expander("Positive Comments", expanded=True):
                                if positive_count > 0:
                                    st.dataframe(
                                        comments_df[comments_df["Sentiment"] == "positive"][["Comment", "Sentiment_Score"]]
                                        .rename(columns={"Sentiment_Score": "Positivity Score"})
                                        .sort_values(by="Positivity Score", ascending=False),
                                        use_container_width=True
                                    )
                                else:
                                    st.info("No positive comments found.")
                            
                            with st.expander("Neutral Comments", expanded=False):
                                if neutral_count > 0:
                                    st.dataframe(
                                        comments_df[comments_df["Sentiment"] == "neutral"][["Comment", "Sentiment_Score"]]
                                        .rename(columns={"Sentiment_Score": "Neutrality Score"}),
                                        use_container_width=True
                                    )
                                else:
                                    st.info("No neutral comments found.")
                            
                            with st.expander("Negative Comments", expanded=False):
                                if negative_count > 0:
                                    st.dataframe(
                                        comments_df[comments_df["Sentiment"] == "negative"][["Comment", "Sentiment_Score"]]
                                        .rename(columns={"Sentiment_Score": "Negativity Score"})
                                        .sort_values(by="Negativity Score"),
                                        use_container_width=True
                                    )
                                else:
                                    st.info("No negative comments found.")
                                    
                        with tab3:
                            # Word Analysis
                            st.subheader("Word Cloud Analysis")
                            
                            if total_comments > 0:
                                # Create three columns for each sentiment wordcloud
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.markdown("### Positive Comments")
                                    if positive_count > 0:
                                        positive_comments = comments_df[comments_df["Sentiment"] == "positive"]["Comment"].tolist()
                                        positive_wordcloud = create_wordcloud(positive_comments, color="#4CAF50")
                                        st.image(positive_wordcloud)
                                    else:
                                        st.info("No positive comments to generate word cloud.")
                                
                                with col2:
                                    st.markdown("### Neutral Comments")
                                    if neutral_count > 0:
                                        neutral_comments = comments_df[comments_df["Sentiment"] == "neutral"]["Comment"].tolist()
                                        neutral_wordcloud = create_wordcloud(neutral_comments, color="#FFC107")
                                        st.image(neutral_wordcloud)
                                    else:
                                        st.info("No neutral comments to generate word cloud.")
                                
                                with col3:
                                    st.markdown("### Negative Comments")
                                    if negative_count > 0:
                                        negative_comments = comments_df[comments_df["Sentiment"] == "negative"]["Comment"].tolist()
                                        negative_wordcloud = create_wordcloud(negative_comments, color="#F44336")
                                        st.image(negative_wordcloud)
                                    else:
                                        st.info("No negative comments to generate word cloud.")
                            else:
                                st.info("No comments available for word cloud generation.")
                                
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
    
    elif analyze_button:
        st.warning("Please enter a YouTube video URL.")
    
    # Display example and information if no URL is entered yet
    if not youtube_url:
        st.info("Enter a YouTube URL above to analyze the sentiment of its comments.")
        
        # Add dashboard element as placeholder
        try:
            response = requests.get("https://pixabay.com/get/gbf126442622b72367ca0e59feed73acd2e0e317da61bade5e24674e287f8ac034d3aa4429dcd087e5a43373f040a1c79d47a330538f7c91d0fa0ae677b25f535_1280.jpg")
            img = Image.open(BytesIO(response.content))
            st.image(img, caption="Sentiment Analysis Dashboard Example", use_column_width=True)
        except Exception as e:
            pass

# Footer information
st.markdown("---")
st.markdown("""
**About this app**: This application demonstrates the use of LSTM neural networks for sentiment analysis.
The model is trained on YouTube comments data to classify sentiments as positive, negative, or neutral.
""")
