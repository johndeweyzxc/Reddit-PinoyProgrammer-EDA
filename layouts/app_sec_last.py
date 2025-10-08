from dash import html


def disclaimer_layout():
    return html.Div([
        html.Div("DISCLAIMER", className="app-subtitle"),
        html.Div("This data analytics dashboard is intended solely for educational and exploratory purposes. The data presented here is sourced from the Pushshift Reddit archive, which collects publicly available subreddit posts. No private messages or user-identifiable information beyond what is publicly accessible on Reddit are included. Please be aware that some posts may contain sensitive, emotional, or personal content, as they reflect the thoughts and experiences shared by users in the subreddit PinoyProgrammer. While the data is public, viewer discretion is advised when exploring certain clusters or visualizations. If you have any concerns, questions, or requests regarding the content or use of this dashboard, please feel free to contact me at johndewey02003@gmail.com.", className="app-label-value")
    ], className="app-section")
