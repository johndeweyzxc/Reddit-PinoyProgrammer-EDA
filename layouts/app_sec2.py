from dash import html


def keywords_per_cluster_layout():
    P1 = "Extracts representative keywords for each cluster using Term Frequency-Inverse Document Frequency (TF-IDF). This method measures how important a term is within a specific cluster compared to all other clusters. In essence, it summarizes each cluster by highlighting terms that make it distinct from the rest."
    return html.Div([
        html.Div("KEYWORDS PER CLUSTER", className="app-subtitle"),
        html.P("Section 2", className="app-label-value-sub"),
        html.Div(P1, className="app-label-value"),
        html.Div(id="cluster-keywords-display", style={
            "flex": "1 1 50%",
            "minWidth": "0"
        }),
    ], className="app-section")
