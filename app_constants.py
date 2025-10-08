import os

HEADER_INFO_PATH = os.path.join("data", "header_info")
OVERVIEW_INFO_PATH = os.path.join("data", "overview_info")
BASIC_COUNTS_PATH = os.path.join("data", "basic_counts")
POSTS_PER_HOUR_PATH = os.path.join("data", "posts_per_hour")
NGRAMS_FREQUENCY_PATH = os.path.join("data", "ngrams_frequency")
WORD_CLOUD_PATH = os.path.join("data", "word_cloud")
POSTS_PER_DAY_PATH = os.path.join("data", "posts_per_day")
SUBSCRIBER_GROWTH_PATH = os.path.join("data", "subscriber_growth")
LATENT_SPACE_PATH = os.path.join("data", "latent_space")
NGRAMS_PER_CLUSTER_PATH = os.path.join("data", "ngrams_per_cluster")
KEYWORDS_PER_CLUSTER_PATH = os.path.join("data", "keywords_per_cluster")
VECTOR_PATH = os.path.join("data", "00_VECTORS")

CLUSTER_NGRAMS_SIZE = 300
COLORS = {
    "primary": "#B1B1A9",
    "secondary": "#506384",
    "primary_main_bg": "#100F14",
    "primary_card_bg": "#1F1F21",
    "primary_text": "#E6E7DF",
    "primary_fill_color": "rgba(80, 99, 132, 0.3)"
}
CHART_LAYOUT = {
    "plot_bgcolor": COLORS["primary_card_bg"],
    "paper_bgcolor": COLORS["primary_card_bg"],
    "font": {"color": COLORS["primary_text"], "family": "ProtoMono"},
    "margin": {"l": 20, "r": 20, "t": 20, "b": 20},
    "xaxis_showgrid": False,
    "yaxis_showgrid": False,
    "xaxis": {
        "zeroline": False
    },
    "yaxis": {
        "zeroline": False
    }
}

INFERENCE_API_URL = "https://johndeweyzxc-sentence-transformer-english-filipino.hf.space/embed"
DEFAULT_SEARCH_TEXT = "I'm feeling depressed and need someone to talk to about my mental health struggles"
