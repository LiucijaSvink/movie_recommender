import json
import requests
from openai import OpenAI
from utils import get_api_key

OPENAI_API_KEY = get_api_key("OPENAI_API_KEY")
TMDB_API_KEY = get_api_key("TMDB_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)


functions = [
    {
        "name": "get_movie_trailer",
        "description": "Fetch a trailer URL for a single movie",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
            },
            "required": ["title"],
        },
    }
]


def get_movie_trailer(title: str) -> str | None:
    """Fetch the trailer URL for a given movie title using TMDb API."""
    search_url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": title,
        "include_adult": "false",
    }
    search_resp = requests.get(search_url, params=params).json()

    if not search_resp.get("results"):
        return None

    movie_id = search_resp["results"][0]["id"]
    videos_url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos"
    videos_resp = requests.get(videos_url, params={"api_key": TMDB_API_KEY}).json()

    def build_url(site: str, key: str) -> str | None:
        if site.lower() == "youtube":
            return f"https://www.youtube.com/watch?v={key}"
        elif site.lower() == "vimeo":
            return f"https://vimeo.com/{key}"
        return None

    for video in videos_resp.get("results", []):
        if video["type"].lower() == "trailer" and video.get("official", False):
            url = build_url(video["site"], video["key"])
            if url:
                return url

    for video in videos_resp.get("results", []):
        if video["type"].lower() == "trailer":
            url = build_url(video["site"], video["key"])
            if url:
                return url

    return None


def run_movie_trailer_search(title: str) -> str | None:
    """Run OpenAI function calling to find a trailer for a movie title."""
    user_message = f"Can you find the trailer for the movie '{title}'?"

    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": user_message}],
            functions=functions,
            function_call="auto",
        )
        message = completion.choices[0].message

        if message.function_call:
            args = json.loads(message.function_call.arguments)
            trailer = get_movie_trailer(args.get("title"))
            return trailer

        return None

    except Exception as e:
        return None
