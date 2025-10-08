from dash import Dash, html, Input, Output

from app_constants import COLORS


def register_latent_space(app: Dash, cluster_color_map: dict):
    @app.callback(
        Output("latent-space-click-info", "children"),
        [Input("latent-space", "clickData")]
    )
    def display_latent_space_click_info(click_data):
        if click_data is None:
            data = {
                "cluster": "None",
                "component_1": "None",
                "component_2": "None",
                "text": "None"
            }
            cluster_color = COLORS["primary_text"]
        else:
            point = click_data["points"][0]
            data = {
                "cluster": point["customdata"][0],
                "component_1": point["x"],
                "component_2": point["y"],
                "text": point["customdata"][1].replace("<br>", " ")
            }
            cluster_color = cluster_color_map[data["cluster"]]
        return html.Div([
            html.Div([
                html.Span("CLUSTER", className="label"),
                html.Span("COMPONENT 1", className="label"),
                html.Span("COMPONENT 2", className="label")
            ], className="text-label", style={"color": cluster_color}),
            html.Div([
                html.Span(str(data["cluster"]), className="value"),
                html.Span(str(data["component_1"]), className="value"),
                html.Span(str(data["component_2"]), className="value")
            ], className="info", style={"color": cluster_color}),
            html.Div("TEXT", className="text-label"),
            html.Div(data["text"], className="text-value")
        ], className="sec1-latent-scatter")
