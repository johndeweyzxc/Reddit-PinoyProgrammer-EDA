import plotly.graph_objects as go
import json
import os

from app_constants import NGRAMS_PER_CLUSTER_PATH
from app_constants import CHART_LAYOUT, COLORS
from app_constants import CLUSTER_NGRAMS_SIZE


def fig_ngrams_per_cluster(subreddit_name: str, cluster_color_map: dict):
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
        layout = CHART_LAYOUT
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
