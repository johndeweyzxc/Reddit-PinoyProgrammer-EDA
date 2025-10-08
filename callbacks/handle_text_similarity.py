from sklearn.metrics.pairwise import cosine_similarity
from dash import Dash, Output, Input, State, callback_context, html
from numpy import ndarray
import numpy as np
from pandas import DataFrame
import pandas as pd
import requests

from app_constants import DEFAULT_SEARCH_TEXT, INFERENCE_API_URL


def create_layout_retrieved_results(df_result: DataFrame, cluster_color_map: dict):
    if df_result is None or len(df_result) == 0:
        return html.Div([
            html.Div("No results found", className="search-result-title")
        ])
    results_components = []
    for _, row in df_result.iterrows():
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
        html.Div(f"SIMILAR POSTS ({len(df_result)} found)",
                 className="search-result-title"),
        html.Div(results_components)
    ])


def encode_sentence(text: str, n_top: int, df_embeddings: DataFrame, embeddings: ndarray):
    print("EXTRACTING INFO: SEMANTIC SIMILARITY")
    res = requests.post(INFERENCE_API_URL, json={"text": text})
    if res.status_code != 200:
        print(f"EXTRACTING INFO: Received {res.status_code} code")
    text_encoded = np.array(res.json()["embedding"]).reshape(1, -1)
    similarities = cosine_similarity(text_encoded, embeddings).flatten()
    indices = similarities.argsort()[-n_top:][::-1]
    result_df = pd.DataFrame([df_embeddings.iloc[i] for i in indices])
    result_df["similarity_score"] = similarities[indices]
    return result_df


def register_text_similarity(app: Dash, cluster_color_map: dict, df_embeddings: DataFrame, embeddings: ndarray):
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
            df_result = encode_sentence(
                text=search_text.strip(),
                n_top=n_results,
                df_embeddings=df_embeddings,
                embeddings=embeddings
            )
            # Create results display
            results_display = create_layout_retrieved_results(
                df_result, cluster_color_map)
            return results_display, search_text
        return html.Div(), search_text if search_text else ""
