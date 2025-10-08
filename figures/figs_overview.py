import plotly.graph_objects as go
from wordcloud import WordCloud
import pandas as pd
import random
import json
import os

from app_constants import POSTS_PER_HOUR_PATH, NGRAMS_FREQUENCY_PATH, WORD_CLOUD_PATH, POSTS_PER_DAY_PATH, SUBSCRIBER_GROWTH_PATH
from app_constants import COLORS, CHART_LAYOUT


def fig_posts_per_hour(subreddit_name: str):
    print("CREATING FIGURE: POSTS PER HOUR")
    df = pd.read_csv(os.path.join(
        POSTS_PER_HOUR_PATH, f"{subreddit_name}.csv"))
    fig = go.Figure(data=[
        go.Bar(
            x=df["hour"],
            y=df["count"],
            text=df["count"],
            textposition="auto",
            marker_color=COLORS["secondary"],
            marker_line_color=COLORS["primary"],
            marker_line_width=1
        )
    ])
    fig.update_layout(CHART_LAYOUT)
    fig.update_xaxes(title="Hour of Day (GMT+8)")
    fig.update_yaxes(title="Number of Posts")
    return fig


def fig_ngrams_frequency(subreddit_name: str):
    print("CREATING FIGURE: N-GRAMS")
    with open(os.path.join(NGRAMS_FREQUENCY_PATH, f"{subreddit_name}.json"), "r") as f:
        data = json.load(f)
    fig = go.Figure(data=[
        go.Bar(
            x=data["frequencies"],
            y=data["texts"],
            text=data["frequencies"],
            textposition="auto",
            orientation="h",
            marker_color=COLORS["secondary"],
            marker_line_color=COLORS["primary"],
            marker_line_width=1
        )
    ])
    fig.update_layout(CHART_LAYOUT)
    fig.update_xaxes(title="Frequency")
    return fig


def fig_word_cloud(subreddit_name: str, most_common_size=30):
    print("CREATING FIGURE: WORD CLOUD")
    with open(os.path.join(WORD_CLOUD_PATH, f"{subreddit_name}.json"), "r") as f:
        data = json.load(f)

    def two_color_func(*args, **kwargs):
        return random.choice(["#9D9D9D", "#F8F0DF", "#FEFBF3", "#79B4B7"])
    wordcloud = WordCloud(
        width=800,
        height=800,
        background_color=COLORS["primary_card_bg"],
        stopwords=None,
        min_font_size=12,
        max_font_size=80,
        color_func=two_color_func,
        relative_scaling=0.5,
        max_words=most_common_size,
        collocations=False,
        prefer_horizontal=0.7
    ).generate_from_frequencies(data)
    wordcloud_array = wordcloud.to_array()
    fig = go.Figure()
    fig.add_trace(go.Image(z=wordcloud_array))
    fig.update_layout(
        xaxis=dict(showgrid=False, showticklabels=False,
                   zeroline=False, visible=False),
        yaxis=dict(showgrid=False, showticklabels=False,
                   zeroline=False, visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=COLORS["primary_card_bg"],
        plot_bgcolor=COLORS["primary_card_bg"],
        showlegend=False,
        hovermode=False
    )
    return fig


def fig_posts_per_day(subreddit_name: str):
    print("CREATING FIGURE: POSTS PER DAY")
    df = pd.read_csv(os.path.join(POSTS_PER_DAY_PATH, f"{subreddit_name}.csv"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["count"],
        mode="lines",
        fill="tonexty",
        fillcolor=COLORS["primary_fill_color"],
        line=dict(color=COLORS["secondary"], width=2),
        name="Posts per Day"
    ))
    fig.update_layout(CHART_LAYOUT)
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Number of Posts")
    return fig


def fig_subscriber_growth(subreddit_name: str):
    print("CREATING FIGURE: SUBSCRIBER GROWTH")
    df = pd.read_csv(os.path.join(
        SUBSCRIBER_GROWTH_PATH, f"{subreddit_name}.csv"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["subscriber"],
        mode="lines",
        fill="tonexty",
        fillcolor=COLORS["primary_fill_color"],
        line=dict(color=COLORS["secondary"], width=2),
        name="Subscriber Growth"
    ))
    fig.update_layout(CHART_LAYOUT)
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Subscribers")
    return fig
