from dash import html, dcc


def overview_layout():
    P1 = "I collected 28,869 text data from the subreddit: PinoyProgrammer from Reddit to explore general trends, patterns, and insights from the community. The data came from the Pushshift archive which collects Reddit submissions and comments. It covers data from September 2014 to December 2024"
    return html.Div([
        html.P("Updated on October 8 2025", className="app-label-value-sub"),
        html.Div("Overview", className="app-subtitle"),
        html.P("Section 0", className="app-label-value-sub"),
        html.P(P1, className="app-label-value"),
        # Posts per hour, N-grams frequency and word cloud
        html.Div([
            html.Div([
                html.Div("POSTS PER HOUR", className="app-chart-title"),
                dcc.Graph(id="posts-per-hour", style={"height": "300px"}),
                html.Div("FIG.4", className="app-figure-number")
            ], className="app-chart-card postsperhour"),
            html.Div([
                html.Div("N-GRAMS FREQUENCY", className="app-chart-title"),
                dcc.Graph(id="ngrams-frequency", style={"height": "300px"}),
                html.Div("FIG.5", className="app-figure-number")
            ], className="app-chart-card postsperhour"),
            html.Div([
                html.Div("WORD CLOUD", className="app-chart-title"),
                dcc.Graph(id="word-cloud", style={"height": "300px"}),
                html.Div("FIG.6", className="app-figure-number")
            ], className="app-chart-card wordcloud"),
        ], className="app-chart-row"),
        # Posts per day and subscriber growth over time
        html.Div([
            html.Div([
                html.Div("POSTS PER DAY", className="app-chart-title"),
                dcc.Graph(id="posts-per-day", style={"height": "300px"}),
                html.Div("FIG.7", className="app-figure-number")
            ], className="app-chart-card"),
            html.Div([
                html.Div("SUBSCRIBER GROWTH OVER TIME",
                         className="app-chart-title"),
                dcc.Graph(id="subscriber-growth", style={"height": "300px"}),
                html.Div("FIG.8", className="app-figure-number")
            ], className="app-chart-card"),
        ], className="app-chart-timeseries"),
    ], className="app-section")
