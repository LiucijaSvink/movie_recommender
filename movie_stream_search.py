import json
import requests
from typing import List, Optional
from openai import OpenAI
from utils import get_api_key, get_country_code

OPENAI_API_KEY = get_api_key("OPENAI_API_KEY")
TMDB_API_KEY = get_api_key("TMDB_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

functions = [
    {
        "name": "get_streaming_services",
        "description": "Find streaming platforms for a movie in a user's country",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "country": {"type": "string"},
            },
            "required": ["title", "country"],
        },
    }
]

def format_providers_list(providers: List[str], movie_title: str, country_name: str) -> Optional[str]:
    """
    Format the list of streaming providers into a readable sentence.
    Returns None if the list is empty.
    """
    if not providers:
        return None

    if len(providers) == 1:
        return f"Available streaming platform in {country_name} for **{movie_title}**: {providers[0]}."

    provider_str = ", ".join(providers[:-1]) + f", and {providers[-1]}"
    return f"Available streaming platforms in {country_name} for **{movie_title}**: {provider_str}."

def get_streaming_services(title: str, country_code: str = "US") -> List[str]:
    """
    Fetch streaming providers for a movie title from TMDb in the given country.
    """
    search_url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": title,
        "include_adult": "false",
    }
    search_resp = requests.get(search_url, params=params).json()

    if not search_resp.get('results'):
        return []

    movie_id = search_resp['results'][0]['id']
    providers_url = f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers"
    providers_resp = requests.get(providers_url, params={"api_key": TMDB_API_KEY}).json()

    country_data = providers_resp.get("results", {}).get(country_code, {})
    all_providers = set()

    for key in ["flatrate", "rent", "buy"]:
        for provider in country_data.get(key, []):
            all_providers.add(provider["provider_name"])

    return list(all_providers)

def run_streaming_search(title: str, user_country_input: str) -> Optional[str]:
    """
    Run a conversation with OpenAI to find streaming platforms for a movie in the user's country.
    Returns a formatted string of providers or None if none found.
    """
    country_code = get_country_code(user_country_input)
    if not country_code:
        return None

    user_message = f"Where can I watch '{title}' if I live in {user_country_input}?"

    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that always calls the 'get_streaming_services' function to find where movies can be streamed."},
            {"role": "user", "content": user_message}
        ]

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            functions=functions,
            function_call={"name": "get_streaming_services"}  # force function call
        )

        message = completion.choices[0].message

        if message.function_call:
            func_args = json.loads(message.function_call.arguments)
            movie_title = func_args["title"]
            user_country = func_args["country"]
            country_code = get_country_code(user_country)

            provider_list = get_streaming_services(movie_title, country_code)
            provider_list_formatted = format_providers_list(provider_list, title, user_country_input)

            return provider_list_formatted

        else:
            return None

    except Exception as e:
        return None