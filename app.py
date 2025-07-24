from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import requests
import random
import json
import dash
import os


HEADER_INFO_PATH = os.path.join("data", "header_info")
OVERVIEW_INFO_PATH = os.path.join("data", "overview_info")
BASIC_COUNTS_PATH = os.path.join("data", "basic_counts")
POSTS_PER_HOUR_PATH = os.path.join("data", "posts_per_hour")
NGRAMS_FREQUENCY_PATH = os.path.join("data", "ngrams_frequency")
WORD_CLOUD_PATH = os.path.join("data", "word_cloud")
POSTS_PER_DAY_PATH = os.path.join("data", "posts_per_day")
SUBSCRIBER_GROWTH_PATH = os.path.join("data", "subscriber_growth")
LATENT_SPACE_PATH = os.path.join("data", "latent_space")
NGRAMS_PER_CLUSTER_PATH = os.path.join("data", "ngrams_per_cluster")
KEYWORDS_PER_CLUSTER_PATH = os.path.join("data", "keywords_per_cluster")

VECTOR_PATH = os.path.join("data", "00_VECTORS")

INFERENCE_API_URL = "https://johndeweyzxc-sentence-transformer-english-filipino.hf.space/embed"

DEFAULT_SEARCH_TEXT = "I'm feeling depressed and need someone to talk to about my mental health struggles"

CLUSTER_NGRAMS_SIZE = 300

COLORS = {
    "primary": "#B1B1A9",
    "secondary": "#506384",
    "primary_main_bg": "#100F14",
    "primary_card_bg": "#1F1F21",
    "primary_text": "#E6E7DF",
    "primary_fill_color": "rgba(80, 99, 132, 0.3)"
}


def import_vectors(subreddit_name: str):
    with open(os.path.join(VECTOR_PATH, f"embeddings_col_map_{subreddit_name}.json"), "r") as f:
        col_map = json.load(f)
        col_map = {idx: col_name for idx,
                   col_name in enumerate(col_map.values())}
    data = np.load(
        os.path.join(VECTOR_PATH, f"embeddings_data_{subreddit_name}.npz"), allow_pickle=True)
    try:
        df = pd.DataFrame(data["metadata"].item())
    except ValueError:
        df = pd.DataFrame(data["metadata"].tolist())
    df.rename(columns=col_map, inplace=True)
    return df, data


app = dash.Dash(__name__)

subreddit_name = ""
subreddit_link = ""
rank_by_size = ""
cluster_color_map = {}
df_vector, data_vector = None, None

app.index_string = f"""
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>"""


def create_search_results_display(results_df):
    if results_df is None or len(results_df) == 0:
        return html.Div([
            html.Div("No results found", className="search-result-title")
        ])
    results_components = []
    for _, row in results_df.iterrows():
        similarity_score = row.get("similarity_score", 0)
        cluster = row.get("cluster", "None")
        text = row.get("corpus", "None")
        result_item = html.Div([
            html.Div([
                html.Div([
                    html.Div("CLUSTER",
                             className="label",
                             style={"color": cluster_color_map[cluster]
                                    }),
                    html.Div(str(cluster),
                             className="search-result-value",
                             style={"color": cluster_color_map[cluster]
                                    })
                ]),
                html.Div([
                    html.Div("SIMILARITY SCORE",
                             className="label",
                             style={"color": cluster_color_map[cluster]
                                    }),
                    html.Div(f"{similarity_score:.3f}",
                             className="search-result-value",
                             style={"color": cluster_color_map[cluster]
                                    }),
                ])
            ], className="search-result-item-header"),
            html.Div("TEXT", className="label"),
            html.Div(text, className="search-result-value")
        ], className="search-result-item")
        results_components.append(result_item)
    return html.Div([
        html.Div(f"SIMILAR POSTS ({len(results_df)} found)",
                 className="search-result-title"),
        html.Div(results_components)
    ])


app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.Div("r/", className="reddit-logo"),
            html.Div([
                html.P("SUBREDDIT DATA ANALYTICS DASHBOARD"),
                html.A(id="subreddit-link", target="_blank"),
            ])
        ], className="header-title"),
        html.Div([
            html.Div([
                html.Div("SUBREDDIT NAME", className="label"),
                html.Div(id="subreddit-name", className="value")
            ]),
            html.Div([
                html.Div("CREATED", className="label"),
                html.Div(id="created", className="value")
            ]),
            html.Div([
                html.Div("RANK BY SIZE", className="label"),
                html.Div(id="rank-by-size", className="value")
            ]),
            html.Div([
                html.Div("DATA RANGE", className="label"),
                html.Div(id="data-range", className="value")
            ]),
        ], className="header-info")
    ], className="header"),

    # Main dashboard container
    html.Div([
        html.Div("Overview", className="section-title"),
        html.Div(id="overview-info", className="text-description"),

        # 1st row: Total posts, subscribers and authors
        html.Div([
            html.Div([
                html.Div("TOTAL POSTS", className="label"),
                html.Div(id="total-posts", className="value"),
                html.Div("FIG.1", className="figure-number")
            ], className="card"),
            html.Div([
                html.Div("SUBSCRIBERS", className="label"),
                html.Div(id="current-subscribers",
                         className="value"),
                html.Div("FIG.2", className="figure-number")
            ], className="card"),
            html.Div([
                html.Div("AUTHORS", className="label"),
                html.Div(id="unique-authors",
                         className="value"),
                html.Div("FIG.3", className="figure-number")
            ], className="card"),
        ], className="overview-metric"),

        # 2nd row: Posts per hour, N-grams frequency and word cloud
        html.Div([
            html.Div([
                html.Div("POSTS PER HOUR", className="chart-title"),
                dcc.Graph(id="posts-per-hour", style={"height": "300px"}),
                html.Div("FIG.4", className="figure-number")
            ], className="chart-card postsperhour"),
            html.Div([
                html.Div("N-GRAMS FREQUENCY", className="chart-title"),
                dcc.Graph(id="ngrams-frequency", style={"height": "300px"}),
                html.Div("FIG.5", className="figure-number")
            ], className="chart-card postsperhour"),
            html.Div([
                html.Div("WORD CLOUD", className="chart-title"),
                dcc.Graph(id="word-cloud", style={"height": "300px"}),
                html.Div("FIG.6", className="figure-number")
            ], className="chart-card wordcloud"),
        ], className="overview-chart"),

        # 3rd: Posts per day and subscriber growth over time
        html.Div([
            html.Div([
                html.Div("POSTS PER DAY", className="chart-title"),
                dcc.Graph(id="posts-per-day", style={"height": "300px"}),
                html.Div("FIG.7", className="figure-number")
            ], className="chart-card"),

            html.Div([
                html.Div("SUBSCRIBER GROWTH OVER TIME",
                         className="chart-title"),
                dcc.Graph(id="subscriber-growth", style={"height": "300px"}),
                html.Div("FIG.8", className="figure-number")
            ], className="chart-card"),
        ], className="overview-chart-timeseries"),

        # 4th row: Latent space visualization
        html.Div("VECTOR REPRESENTATION AND PROJECTION",
                 className="section-title"),
        html.Div(
            "Vector representation refers to the process of converting raw data like text or images into numerical vectors that a model can understand and work with. These vectors capture meaningful patterns and features from the data. Projection is the technique of mapping these high-dimensional vectors into a lower-dimensional space. This is often done to simplify the data, highlight important structures, or visualize it in 2D or 3D space.",
            className="text-description"
        ),
        html.Div([
            html.Div("LATENT SPACE VISUALIZATION",
                     className="chart-title"),
            dcc.Graph(id="latent-space",
                      style={"height": "500px"}),
            html.Div("FIG.9", className="figure-number"),
        ], className="latent-card"),
        html.Div(id="latent-space-click-info"),
        html.Div(
            "Posts with similar meanings or topics appear closer together in the visualization. For example, subreddit posts like \"I've been cheating with my long-term boyfriend...\" and \"Talamak na cheating sa top BPO here in Manila...\" will be positioned near each other when plotted in a 2D space. This is because the system groups them based on shared themes.",
            className="text-description"
        ),
        html.Div(
            "A sample of 15,000 subreddit post is collected from the population. Each subreddit post (based on its title and text) is converted into a list of numbers called a sentence embedding. This is done using a Sentence Transformer, a model that captures the meaning of the post in numerical form. Posts with similar meanings end up with embeddings that are close to each other in high-dimensional space. To group these similar posts, K-Means clustering was applied with the number of clusters set to 5. This groups posts into five distinct categories based on content similarity. To make the data easier to explore, dimensionality reduction techniques like PCA (Principal Component Analysis), t-SNE, and UMAP is applied to project the high-dimensional embeddings into 2D. In the visualization above, each point represents a post, and its color shows the cluster it belongs to. The fact that K-Means clusters remain visually separated in 2D provides supporting evidence that the original high-dimensional vectors capture meaningful, separable patterns and that the dimensionality reduction retains that structure well enough for visualization.",
            className="text-description"
        ),

        # 5th row: TFIDF per cluster
        html.Div("KEYWORDS PER CLUSTER", className="section-title"),
        html.Div(
            "Extracts representative keywords for each cluster using Term Frequency-Inverse Document Frequency (TF-IDF). This method measures how important a term is within a specific cluster compared to all other clusters. In essence, it summarizes each cluster by highlighting terms that make it distinct from the rest.",
            className="text-description"
        ),
        html.Div(id="cluster-keywords-display", style={
            "flex": "1 1 50%",
            "minWidth": "0"
        }),

        # 6th row: Semantic searching
        html.Div("SEMANTIC SEARCHING", className="section-title"),
        html.Div(
            "Semantic search goes beyond just matching keywords—it understands the meaning and context behind a user's search. Each subreddit post (based on its text and title) is converted into a vector that captures its overall message. This allows the system to find and rank posts that are similar in meaning, even if they don’t use the exact same words.",
            className="text-description"
        ),
        html.Div([
            # Search input
            dcc.Textarea(
                id="search-input",
                placeholder="Enter text to find similar posts...",
                className="search-input",
                value=""
            ),
            # Search controls
            html.Div([
                html.Div([
                    html.Button(
                        "SEARCH",
                        id="search-button",
                        className="button primary",
                        n_clicks=0
                    ),
                    html.Button(
                        "DEFAULT",
                        id="default-button",
                        className="button secondary",
                        n_clicks=0
                    )
                ], className="buttons"),
                html.Div([
                    html.Label("Results:", className="label"),
                    dcc.Dropdown(
                        id="results-count-dropdown",
                        options=[
                            {"label": "5", "value": 5},
                            {"label": "10", "value": 10},
                            {"label": "15", "value": 15},
                            {"label": "20", "value": 20}
                        ],
                        value=10,
                        className="dropdown",
                        clearable=False
                    )
                ], className="result-quantity")
            ], className="search-controls"),

            # Results container
            html.Div(id="search-results-container",
                     className="search-result")
        ], className="semantic-search"),

        # 7th row: N-grams per cluster
        html.Div("N-GRAMS PER CLUSTER", className="section-title"),
        html.Div(
            "An n-gram is a sequence of n words that appear together in a sentence. For example, a bigram (n=2) might be \"mental health\", and a trigram (n=3) could be \"I feel lost\". In this dashboard, I analyzed the most frequent n-grams within each cluster to highlight common phrases or themes used by people in similar types of posts.",
            className="text-description"
        ),
        html.Div([
            html.Div(id="ngrams-cluster-container", className="charts"),
        ], className="ngrams-cluster"),

        # Disclaimer
        html.Div("DISCLAIMER", className="section-title"),
        html.Div("This data analytics dashboard is intended solely for educational and exploratory purposes. The data presented here is sourced from the Pushshift Reddit archive, which collects publicly available subreddit posts. No private messages or user-identifiable information beyond what is publicly accessible on Reddit are included. Please be aware that some posts may contain sensitive, emotional, or personal content, as they reflect the thoughts and experiences shared by users in the subreddit r/OffMyChestPH. While the data is public, viewer discretion is advised when exploring certain clusters or visualizations. If you have any concerns, questions, or requests regarding the content or use of this dashboard, please feel free to contact me at johndewey02003@gmail.com.", className="text-description")
    ], className="dashboard-container"),

    # Data store for semantic search
    html.Div(id="semantic-search-data-store", style={"display": "none"}),

    # Footer
    html.Div([
        html.Div([
            html.Div("Subreddit Data Analytics Dashboard 📊",
                     className="subreddit-data-analytics-dashboard"),
            html.Div("Designed by John Dewey 🛰️",
                     className="designed-by-john-dewey")
        ], className="footer-branding"),

        html.Div([
            html.Div("Connect with me 🌐", className="connect-with-me"),
            html.Div([
                html.A("LinkedIn", target="_blank", href="https://www.linkedin.com/in/john-dewey-047066344/",
                       className="links",
                       style={"marginRight": "1rem"}),
                html.A("Github", target="_blank", href="https://github.com/johndeweyzxc",
                       className="links"),
            ], style={
                "display": "flex",
            })
        ], className="footer-connect"),

    ], className="footer")
])


def get_chart_layout():
    return {
        "plot_bgcolor": COLORS["primary_card_bg"],
        "paper_bgcolor": COLORS["primary_card_bg"],
        "font": {"color": COLORS["primary_text"], "family": "ProtoMono"},
        "margin": {"l": 20, "r": 20, "t": 20, "b": 20},
        "xaxis_showgrid": False,
        "yaxis_showgrid": False,
        "xaxis": {
            "zeroline": False
        },
        "yaxis": {
            "zeroline": False
        }
    }


def header_info(subreddit_name: str):
    print("EXTRACTING INFO: HEADER INFO")
    with open(os.path.join(HEADER_INFO_PATH, f"{subreddit_name}.json"), "r") as f:
        data = json.load(f)
    return (data["created"], data["data_range"])


def date_overview_info(subreddit_name: str):
    print("EXTRACTING INFO: DATA RANGE OVERVIEW")
    with open(os.path.join(OVERVIEW_INFO_PATH, f"{subreddit_name}.json"), "r") as f:
        data = json.load(f)
    return data["data_head_date"], data["data_tail_date"]


def row1_figs(subreddit_name: str):
    print("CREATING FIGURE: BASIC COUNTS")
    with open(os.path.join(BASIC_COUNTS_PATH, f"{subreddit_name}.json"), "r") as f:
        data = json.load(f)
    return data["total_posts"], data["subscribers"], data["authors"]


def posts_per_hour_fig(subreddit_name: str):
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
    fig.update_layout(get_chart_layout())
    fig.update_xaxes(title="Hour of Day (GMT+8)")
    fig.update_yaxes(title="Number of Posts")
    return fig


def ngrams_frequency_fig(subreddit_name: str):
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
    fig.update_layout(get_chart_layout())
    fig.update_xaxes(title="Frequency")
    return fig


def word_cloud_fig(subreddit_name: str, most_common_size=30):
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


def posts_per_day_fig(subreddit_name: str):
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
    fig.update_layout(get_chart_layout())
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Number of Posts")
    return fig


def subscriber_growth_fig(subreddit_name: str):
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
    fig.update_layout(get_chart_layout())
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Subscribers")
    return fig


def latent_space_fig(subreddit_name: str):
    global cluster_color_map
    print("CREATING FIGURE: LATENT SPACE")
    df = pd.read_csv(os.path.join(LATENT_SPACE_PATH, f"{subreddit_name}.csv"))
    fig = px.scatter(
        data_frame=df,
        x=df["component_1"],
        y=df["component_2"],
        color=df["cluster"].astype(str),
        hover_data={
            "cluster": True,
            "text": df["corpus_wrapped"]
        },
        labels={
            "component_1": "Component 1",
            "component_2": "Component 2",
            "color": "Cluster",
            "text": "Text"
        },
        opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Set3)
    layout = get_chart_layout()
    layout.update({
        "xaxis_title": "Component 1",
        "yaxis_title": "Component 2",
        "legend": {
            "title": "Cluster",
            "font": {"color": COLORS["primary_text"]},
            "bgcolor": "rgba(0,0,0,0)"
        }
    })
    fig.update_layout(layout)
    # Extract HEX value of a color assigned on a cluster
    for trace in fig.data:
        if "marker" in trace and "color" in trace.marker:
            cluster = int(trace.name) if trace.name.isdigit() else trace.name
            color = trace.marker.color
            cluster_color_map[cluster] = color
    return fig


def ngrams_per_cluster_fig(subreddit_name: str):
    print("CREATING FIGURE: CLUSTER NGRAMS")
    with open(os.path.join(NGRAMS_PER_CLUSTER_PATH, f"{subreddit_name}.json"), "r") as f:
        data = json.load(f)
    clusters = list(data.keys())
    cluster_figures = []
    for cluster_id in clusters:
        texts = data[cluster_id]["texts"]
        frequencies = data[cluster_id]["frequencies"]
        fig = go.Figure(data=[
            go.Bar(
                x=frequencies,
                y=texts,
                text=frequencies,
                textposition="auto",
                orientation="h",
                marker_color=cluster_color_map[int(cluster_id)],
                marker_line_color=COLORS["primary"],
                marker_line_width=1
            )
        ])
        layout = get_chart_layout()
        layout.update({
            "xaxis_title": "Frequency",
            "yaxis_title": "N-grams",
            "height": CLUSTER_NGRAMS_SIZE
        })
        fig.update_layout(layout)
        cluster_figures.append({
            "cluster_id": cluster_id,
            "figure": fig
        })
    return cluster_figures


def keywords_per_cluster(subreddit_name: str):
    print("EXTRACTING INFO: CLUSTER KEYWORD")
    with open(os.path.join(KEYWORDS_PER_CLUSTER_PATH, f"{subreddit_name}.json"), "r") as f:
        data = json.load(f)
    return data


def create_cluster_keywords_display(keywords: dict):
    cluster_elements = []
    for cluster_id in sorted(keywords.keys()):
        texts = keywords[cluster_id]
        keywords_text = " ".join(texts)
        cluster_element = html.Div([
            html.Div(f"CLUSTER {cluster_id}", className="title"),
            html.Div(keywords_text, className="text")
        ], className="card", style={
            "color": cluster_color_map[int(cluster_id)],
        })
        cluster_elements.append(cluster_element)
    return html.Div(cluster_elements, className="keywords")


def find_similar_posts(text: str, n_top: int):
    print("EXTRACTING INFO: SEMANTIC SIMILARITY")
    embeddings = data_vector["embeddings"]
    res = requests.post(INFERENCE_API_URL, json={"text": text})
    if res.status_code != 200:
        print(f"EXTRACTING INFO: Received {res.status_code} code")
    text_encoded = np.array(res.json()["embedding"]).reshape(1, -1)
    similarities = cosine_similarity(text_encoded, embeddings).flatten()
    indices = similarities.argsort()[-n_top:][::-1]
    result_df = pd.DataFrame([df_vector.iloc[i] for i in indices])
    result_df["similarity_score"] = similarities[indices]
    return result_df


@app.callback(
    [Output("search-results-container", "children"),
     Output("search-input", "value")],
    [Input("search-button", "n_clicks"),
     Input("default-button", "n_clicks")],
    [State("search-input", "value"),
     State("results-count-dropdown", "value")]
)
def handle_search(search_clicks, default_clicks, search_text, n_results):
    ctx = callback_context
    if not ctx.triggered:
        return html.Div(), ""
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    # Handle default button click
    if button_id == "default-button":
        return html.Div(), DEFAULT_SEARCH_TEXT
    # Handle search button click
    if button_id == "search-button" and search_text and search_text.strip():
        try:
            results_df = find_similar_posts(
                text=search_text.strip(),
                n_top=n_results
            )
            # Create results display
            results_display = create_search_results_display(results_df)
            return results_display, search_text
        except Exception as e:
            error_display = html.Div([
                html.Div("Error occurred during search",
                         className="search-result-title"),
                html.Div(f"Error: {str(e)}", className="search-result-value")
            ])
            return error_display, search_text
    return html.Div(), search_text if search_text else ""


@app.callback(
    Output("latent-space-click-info", "children"),
    [Input("latent-space", "clickData")]
)
def display_latent_space_click_info(click_data):
    if click_data is None:
        data = {
            "cluster": "None",
            "component_1": "None",
            "component_2": "None",
            "text": "None"
        }
        cluster_color = COLORS["primary_text"]
    else:
        point = click_data["points"][0]
        data = {
            "cluster": point["customdata"][0],
            "component_1": point["x"],
            "component_2": point["y"],
            "text": point["customdata"][1].replace("<br>", " ")
        }
        cluster_color = cluster_color_map[data["cluster"]]
    return html.Div([
        html.Div([
            html.Span("CLUSTER", className="label"),
            html.Span("COMPONENT 1", className="label"),
            html.Span("COMPONENT 2", className="label")
        ], style={"fontWeight": "bold", "marginBottom": "0.25rem", "color": cluster_color}),
        html.Div([
            # TODO: Fix CSS style width for small screen size (Galaxy Z Fold 5)
            html.Span(str(data["cluster"]), className="value"),
            html.Span(str(data["component_1"]), className="value"),
            html.Span(str(data["component_2"]), className="value")
        ], className="info", style={"color": cluster_color}),
        html.Div("TEXT", className="text-label"),
        html.Div(data["text"], className="text-value")
    ], className="latent-scatter")


@app.callback(
    [
        Output("subreddit-link", "children"),
        Output("subreddit-link", "href"),
        Output("subreddit-name", "children"),
        Output("created", "children"),
        Output("rank-by-size", "children"),
        Output("data-range", "children"),
        Output("overview-info", "children"),
        Output("total-posts", "children"),
        Output("current-subscribers", "children"),
        Output("unique-authors", "children"),
        Output("posts-per-hour", "figure"),
        Output("ngrams-frequency", "figure"),
        Output("word-cloud", "figure"),
        Output("posts-per-day", "figure"),
        Output("subscriber-growth", "figure")],
    Output("latent-space", "figure"),
    Output("cluster-keywords-display", "children"),
    Output("ngrams-cluster-container", "children"),
    [Input("semantic-search-data-store", "children")]
)
def update_dashboard(data_store):
    print("\n--- UI UPDATE ---")

    created, data_range = header_info(subreddit_name)
    date_start, date_end = date_overview_info(subreddit_name)
    overview_info = html.Span([
        "This dashboard presents a social media data analysis of the subreddit ",
        html.A(f"r/{subreddit_name}", href=subreddit_link, target="_blank", style={
            "textDecoration": "underline",
            "color": COLORS["primary_text"],
        }),
        ", based on posts from ",
        html.A("Reddit", href="https://www.reddit.com/", target="_blank", style={
            "textDecoration": "underline",
            "color": COLORS["primary_text"]
        }),
        ". The data was collected from the ",
        html.A("Pushshift archive", href="https://academictorrents.com/details/1614740ac8c94505e4ecb9d88be8bed7b6afddd4", target="_blank", style={
            "textDecoration": "underline",
            "color": COLORS["primary_text"]
        }),
        " and covers activity from ",
        f"{date_start} to {date_end}. ",
        "All raw data went through a custom-built pipeline for cleaning, processing, and visualization to uncover trends, patterns, and insights from the community."
    ], style={
        "fontSize": ".85rem"
    })
    total_posts, subscribers, authors = row1_figs(subreddit_name)
    posts_hour_fig = posts_per_hour_fig(subreddit_name)
    ngrams_fig = ngrams_frequency_fig(subreddit_name)
    word_fig = word_cloud_fig(subreddit_name)
    posts_day_fig = posts_per_day_fig(subreddit_name)
    sub_growth_fig = subscriber_growth_fig(subreddit_name)
    late_space_fig = latent_space_fig(subreddit_name)
    keywords_dict = keywords_per_cluster(subreddit_name)
    cluster_keywords_display = create_cluster_keywords_display(keywords_dict)
    clus_ngrams_figs = ngrams_per_cluster_fig(subreddit_name)

    cluster_ngrams_components = []
    for cluster_data in clus_ngrams_figs:
        cluster_id = cluster_data["cluster_id"]
        figure = cluster_data["figure"]
        cluster_component = html.Div([
            html.Div(f"CLUSTER {cluster_id}", className="chart-title"),
            dcc.Graph(
                figure=figure,
                style={"height": f"{CLUSTER_NGRAMS_SIZE}px"}
            ),
            html.Div(f"FIG.{10 + int(cluster_id)}", className="figure-number")
        ], className="chart")
        cluster_ngrams_components.append(cluster_component)

    return (
        subreddit_link, subreddit_link, subreddit_name, created, rank_by_size, data_range,
        overview_info,
        total_posts, subscribers, authors,
        posts_hour_fig, ngrams_fig, word_fig,
        posts_day_fig, sub_growth_fig,
        late_space_fig,
        cluster_keywords_display,
        cluster_ngrams_components)


if __name__ == "__main__":
    subreddit_name = "PinoyProgrammer"
    subreddit_link = "https://www.reddit.com/r/PinoyProgrammer/"
    rank_by_size = "TOP 1%"
    df_vector, data_vector = import_vectors(subreddit_name)

    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)
    # NOTE: Change to:
    port = int(os.environ.get("PORT", 8050))  # default for local
    app.run(debug=False, host="0.0.0.0", port=port)
    # When pushing into huggingface space
    # app.run(debug=True)
