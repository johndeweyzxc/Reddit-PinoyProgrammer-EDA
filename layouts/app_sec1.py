from dash import html, dcc


def latent_space_layout():
    P1 = "Latent space visualization is a scatter plot data visulization where points that are conceptually similar are located closer in high-dimensional latent space. To visualize this space, I used UMAP to convert the high-dimensional space to 2 dimension. K-Means clustering was applied in the original high-dimensional latent space. The fact that the resulting cluster labels remain visibly separated in the 2D UMAP projection suggests that the original embeddings contain meaningful and separable patterns, and that UMAP preserved those patterns well enough for effective visualization."
    P2 = "Each data point is a text where semantically similar text appear closer together in the visualization."
    return html.Div([
        html.Div("LATENT SPACE",
                 className="app-subtitle"),
        html.P("Section 1", className="app-label-value-sub"),
        html.Div(P1, className="app-label-value"),
        html.Div([
            html.Div("LATENT SPACE VISUALIZATION",
                     className="app-chart-title"),
            dcc.Graph(id="latent-space",
                      style={"height": "500px"}),
            html.Div("FIG.9", className="app-figure-number"),
        ], className="sec1-latent-card"),
        html.Div(id="latent-space-click-info"),
        html.Div(P2, className="app-label-value")
    ], className="app-section")
