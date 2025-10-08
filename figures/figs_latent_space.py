import plotly.express as px
import pandas as pd
import os

from app_constants import LATENT_SPACE_PATH
from app_constants import CHART_LAYOUT, COLORS


def fig_latent_space(subreddit_name: str):
    cluster_color_map = {}
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
    layout = CHART_LAYOUT
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
    return fig, cluster_color_map
