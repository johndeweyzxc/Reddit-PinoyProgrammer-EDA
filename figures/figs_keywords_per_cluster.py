from dash import html


def display_keywords_per_cluster(keywords: dict, cluster_color_map: dict):
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
