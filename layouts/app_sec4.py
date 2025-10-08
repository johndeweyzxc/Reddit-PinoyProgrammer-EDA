from dash import html


def ngrams_per_cluster_layout():
    P1 = "An n-gram is a sequence of n words that appear together in a sentence. For example, a bigram (n=2) might be \"mental health\", and a trigram (n=3) could be \"I feel lost\". In this dashboard, I analyzed the most frequent n-grams within each cluster to highlight common phrases or themes used by people in similar types of posts."
    return html.Div([
        html.Div("N-GRAMS PER CLUSTER", className="app-subtitle"),
        html.P("Section 4", className="app-label-value-sub"),
        html.Div(P1, className="app-label-value"),
        html.Div([
            html.Div(id="ngrams-cluster-container", className="charts"),
        ], className="ngrams-cluster"),
    ], className="app-section")
