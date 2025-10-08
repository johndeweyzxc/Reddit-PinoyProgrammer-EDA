from dash import html, dcc


def semantic_search_layout():
    P1 = "Semantic search goes beyond just matching keywords—it understands the meaning and context behind a user's search. Each subreddit post (based on its text and title) is converted into a vector that captures its overall message. This allows the system to find and rank posts that are similar in meaning, even if they don't use the exact same words."
    return html.Div([
        html.Div("SEMANTIC SEARCHING", className="app-subtitle"),
        html.P("Section 3", className="app-label-value-sub"),
        html.Div(P1, className="app-label-value"),
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
            dcc.Loading(
                id="loading-spinner-on-get-similar-images",
                type="default",
                children=html.Div(id="search-results-container",
                                  className="search-result")
            ),
        ], className="semantic-search"),
    ], className="app-section")
