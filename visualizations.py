import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
from PIL import Image

def create_sentiment_pie_chart(sentiment_counts):
    """
    Create a pie chart for sentiment distribution.
    
    Parameters:
    sentiment_counts (DataFrame): DataFrame with sentiment counts
    
    Returns:
    Figure: Plotly pie chart figure
    """
    # Define colors matching the application theme
    colors = {
        'positive': '#4CAF50',  # Green
        'neutral': '#FFC107',   # Amber
        'negative': '#F44336'   # Red
    }
    
    # Create color list based on sentiments
    color_list = [colors.get(sentiment, '#7b2cbf') for sentiment in sentiment_counts['Sentiment']]
    
    # Create pie chart
    fig = px.pie(
        sentiment_counts, 
        values='Count', 
        names='Sentiment',
        color='Sentiment',
        color_discrete_map={
            'positive': colors['positive'],
            'neutral': colors['neutral'],
            'negative': colors['negative']
        },
        title='Sentiment Distribution'
    )
    
    # Update layout
    fig.update_layout(
        font=dict(family="Roboto, Open Sans, sans-serif"),
        legend_title_text='Sentiment',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Update traces
    fig.update_traces(
        textinfo='percent+label',
        pull=[0.05, 0, 0],
        marker=dict(line=dict(color='#ffffff', width=1))
    )
    
    return fig

def create_sentiment_bar_chart(sentiment_counts):
    """
    Create a bar chart for sentiment distribution.
    
    Parameters:
    sentiment_counts (DataFrame): DataFrame with sentiment counts
    
    Returns:
    Figure: Plotly bar chart figure
    """
    # Define colors matching the application theme
    colors = {
        'positive': '#4CAF50',  # Green
        'neutral': '#FFC107',   # Amber
        'negative': '#F44336'   # Red
    }
    
    # Create color list based on sentiments
    color_list = [colors.get(sentiment, '#7b2cbf') for sentiment in sentiment_counts['Sentiment']]
    
    # Create bar chart
    fig = px.bar(
        sentiment_counts, 
        x='Sentiment', 
        y='Count',
        color='Sentiment',
        color_discrete_map={
            'positive': colors['positive'],
            'neutral': colors['neutral'],
            'negative': colors['negative']
        },
        title='Sentiment Counts'
    )
    
    # Update layout
    fig.update_layout(
        font=dict(family="Roboto, Open Sans, sans-serif"),
        xaxis_title='Sentiment',
        yaxis_title='Count',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_wordcloud(texts, color='white', width=400, height=200):
    """
    Create a word cloud from a list of texts.
    
    Parameters:
    texts (list): List of text samples
    color (str): Color map for the word cloud
    width (int): Width of the image
    height (int): Height of the image
    
    Returns:
    BytesIO: Image of the word cloud
    """
    # Combine texts
    if not texts:
        return None
        
    text = ' '.join(texts)
    
    # Create WordCloud object
    wc = WordCloud(
        width=width, 
        height=height, 
        background_color='white',
        colormap=color,
        max_words=100,
        contour_width=1,
        contour_color='black'
    )
    
    # Generate word cloud
    wc.generate(text)
    
    # Plot the word cloud
    plt.figure(figsize=(width/100, height/100), dpi=100)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    
    # Save to BytesIO
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()
    
    return buf

def create_sentiment_trend_chart(sentiment_data):
    """
    Create a line chart showing sentiment trends over time.
    
    Parameters:
    sentiment_data (DataFrame): DataFrame with sentiment data over time
    
    Returns:
    Figure: Plotly line chart figure
    """
    # Create line chart
    fig = px.line(
        sentiment_data,
        x='timestamp',
        y=['positive', 'neutral', 'negative'],
        title='Sentiment Trend Over Time',
        labels={'timestamp': 'Time', 'value': 'Count'}
    )
    
    # Update layout
    fig.update_layout(
        font=dict(family="Roboto, Open Sans, sans-serif"),
        xaxis_title='Time',
        yaxis_title='Count',
        legend_title='Sentiment',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Update line colors
    fig.update_traces(
        line=dict(width=2),
        selector=dict(name='positive'),
        line_color='#4CAF50'
    )
    
    fig.update_traces(
        line=dict(width=2),
        selector=dict(name='neutral'),
        line_color='#FFC107'
    )
    
    fig.update_traces(
        line=dict(width=2),
        selector=dict(name='negative'),
        line_color='#F44336'
    )
    
    return fig
