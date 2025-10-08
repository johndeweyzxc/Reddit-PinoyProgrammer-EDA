from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
import logging
import json
import dash
import os

from layouts.app_sec0 import overview_layout
from layouts.app_sec1 import latent_space_layout
from layouts.app_sec2 import keywords_per_cluster_layout
from layouts.app_sec3 import semantic_search_layout
from layouts.app_sec4 import ngrams_per_cluster_layout
from layouts.app_sec_last import disclaimer_layout

from figures.figs_overview import *
from figures.figs_ngrams_per_cluster import *
from figures.figs_keywords_per_cluster import *
from figures.figs_latent_space import *

from callbacks.handle_text_similarity import register_text_similarity
from callbacks.handle_latent_space import register_latent_space

from app_constants import KEYWORDS_PER_CLUSTER_PATH, VECTOR_PATH
from app_constants import CLUSTER_NGRAMS_SIZE


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
    return df, data["embeddings"]


subreddit_name = "PinoyProgrammer"
subreddit_link = "https://www.reddit.com/r/PinoyProgrammer/"
rank_by_size = "TOP 1%"
df_embeddings, embeddings = import_vectors(subreddit_name)

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

app = dash.Dash(__name__)
app.title = "Reddit Exploratory Data Analysis of Subreddit: OffMyChestPH"
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


latent_space_fig, cluster_color_map = fig_latent_space(subreddit_name)


app.layout = html.Div([
    # HEADER
    html.Div([
        html.Div([
            html.H1("REDDIT EXPLORATORY DATA ANALYSIS",
                    className="app-subtitle",
                    style={"margin": "0"})
        ]),
        html.Div([
            html.A("SOURCE DATASET",
                   href="https://academictorrents.com/details/ba051999301b109eab37d16f027b3f49ade2de13",
                   target="_blank",
                   className="header-source-dataset-link")
        ])
    ], className="header"),
    # MAIN CONTENT
    html.Div([
        # SECTION 0: OVERVIEW
        overview_layout(),
        # SECTION 1: LATENT SPACE
        latent_space_layout(),
        # SECTION 2: KEYWORDS PER CLUSTER
        keywords_per_cluster_layout(),
        # SECTION 3: SEMANTIC SEARCH
        semantic_search_layout(),
        # SECTION 4: N-GRAMS PER CLUSTER
        ngrams_per_cluster_layout(),
        # SECTION LAST: DISCLAIMER
        disclaimer_layout(),
        # Data store for semantic search
        html.Div(id="semantic-search-data-store", style={"display": "none"}),
    ], className="app-container"),
    # FOOTER
    html.Div([
        html.Div([
            html.Div("REDDIT EXPLORATORY DATA ANALYSIS 📊",
                     className="footer-title"),
            html.Div("By John Dewey 🚀", className="footer-subtitle")]),
        html.Div([
            html.Div("CONNECT WITH ME 🌐", className="footer-title"),
            html.Div([
                html.A("My LinkedIn", target="_blank", href="https://www.linkedin.com/in/john-dewey-altura-047066344/",
                       className="footer-links",
                       style={"marginRight": "1rem"}),
                html.A("My Github", target="_blank", href="https://github.com/johndeweyzxc",
                       className="footer-links"),
            ], className="footer-connect-with-me")
        ]),
        html.Div([
            html.Div("MY OTHER PROJECTS 🔬", className="footer-title"),
            html.Div([
                html.A("Philippines Exploratory Data Analysis", target="_blank", href="https://johndeweyzxc-project-redirector.hf.space/Reddit-Philippines-EDA",
                       className="footer-links",
                       style={"marginRight": "1rem"}),
                html.A("OffMyChestPH Exploratory Data Analysis", target="_blank", href="https://johndeweyzxc-project-redirector.hf.space/Reddit-OffMyChestPH-EDA",
                       className="footer-links"),
            ], className="footer-connect-with-me")
        ]),
        html.Div([
            html.Div("EMAIL ✉️",
                     className="footer-title"),
            html.Div("johndewey02003@gmail.com",
                     className="footer-subtitle")
        ]),
    ], className="footer")
])


@app.callback(
    Output("posts-per-hour", "figure"),
    Output("ngrams-frequency", "figure"),
    Output("word-cloud", "figure"),
    Output("posts-per-day", "figure"),
    Output("subscriber-growth", "figure"),
    Output("latent-space", "figure"),
    Output("cluster-keywords-display", "children"),
    Output("ngrams-cluster-container", "children"),
    [Input("semantic-search-data-store", "children")]
)
def update_dashboard(data_store):
    def keywords_per_cluster(subreddit_name: str):
        print("EXTRACTING INFO: CLUSTER KEYWORD")
        with open(os.path.join(KEYWORDS_PER_CLUSTER_PATH, f"{subreddit_name}.json"), "r") as f:
            data = json.load(f)
        return data

    def ngrams_per_cluster(clus_ngrams_fig: list):
        cluster_ngrams_components = []
        for cluster_data in clus_ngrams_figs:
            cluster_id = cluster_data["cluster_id"]
            figure = cluster_data["figure"]
            cluster_component = html.Div([
                html.Div(f"CLUSTER {cluster_id}", className="app-chart-title"),
                dcc.Graph(
                    figure=figure,
                    style={"height": f"{CLUSTER_NGRAMS_SIZE}px"}
                ),
                html.Div(f"FIG.{10 + int(cluster_id)}",
                         className="app-figure-number")
            ], className="chart")
            cluster_ngrams_components.append(cluster_component)
        return cluster_ngrams_components
    print("\n--- UI UPDATE ---")

    posts_hour_fig = fig_posts_per_hour(subreddit_name)
    ngrams_fig = fig_ngrams_frequency(subreddit_name)
    word_fig = fig_word_cloud(subreddit_name)
    posts_day_fig = fig_posts_per_day(subreddit_name)
    sub_growth_fig = fig_subscriber_growth(subreddit_name)
    keywords_dict = keywords_per_cluster(subreddit_name)
    cluster_keywords_display = display_keywords_per_cluster(
        keywords_dict, cluster_color_map)
    clus_ngrams_figs = fig_ngrams_per_cluster(
        subreddit_name, cluster_color_map)
    cluster_ngrams_components = ngrams_per_cluster(clus_ngrams_figs)

    return (
        posts_hour_fig,
        ngrams_fig,
        word_fig,
        posts_day_fig,
        sub_growth_fig,
        latent_space_fig,
        cluster_keywords_display,
        cluster_ngrams_components)


if __name__ == "__main__":
    register_text_similarity(
        app=app,
        cluster_color_map=cluster_color_map,
        df_embeddings=df_embeddings,
        embeddings=embeddings
    )
    register_latent_space(app, cluster_color_map)
    # NOTE: Testing
    # app.run(debug=True)
    # NOTE: Production
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)
