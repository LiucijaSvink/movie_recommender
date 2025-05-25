import json
import requests
from typing import List, Dict, Optional, Union
from openai import OpenAI
from utils import get_api_key

OPENAI_API_KEY: str = get_api_key("OPENAI_API_KEY")
TMDB_API_KEY: str = get_api_key("TMDB_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

functions = [
    {
        "name": "get_movie_ratings",
        "description": "Fetch TMDb ratings for movies and return the top 3",
        "parameters": {
            "type": "object",
            "properties": {
                "movies": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["title", "reason"],
                    },
                    "description": "List of movies with titles and reasons",
                }
            },
            "required": ["movies"],
        },
    }
]

def get_movie_rating(title: str) -> Dict[str, Optional[Union[str, float]]]:
    """Get movie rating from TMDb for a given title."""
    url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": title,
        "include_adult": "false",
    }

    response = requests.get(url, params=params)

    try:
        data = response.json()
    except Exception as e:
        return {"title": title, "rating": None}

    if response.status_code != 200 or "results" not in data or not data["results"]:
        return {"title": title, "rating": None}

    movie = data["results"][0]
    rating = movie.get("vote_average")

    return {"title": title, "rating": rating}

def get_movie_ratings(movies: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Get ratings for a list of movies and return the top 3 by rating."""
    results = []
    for movie in movies:
        rating_info = get_movie_rating(movie["title"])
        results.append({
            "title": movie["title"],
            "reason": movie["reason"],
            "rating": rating_info.get("rating")
        })

    # Sort descending by rating, None values last
    sorted_movies = sorted(
        results,
        key=lambda x: (x['rating'] is not None, x['rating']),
        reverse=True
    )

    top3_movies = [{"title": m["title"], "reason": m["reason"]} for m in sorted_movies[:3]]
    return top3_movies

def run_movie_rating_search(
    movies_with_reasons: List[Union[Dict[str, str], object]]
) -> List[Dict[str, str]]:
    """
    Accepts list of MovieRecommendation objects or dicts with 'title' and 'reason'.
    Uses OpenAI function calling to fetch TMDb ratings and return top 3 results.
    """
    # Convert pydantic models to dicts if needed
    if len(movies_with_reasons) > 0 and hasattr(movies_with_reasons[0], "title"):
        movies_with_reasons = [
            {"title": m.title, "reason": m.reason} for m in movies_with_reasons
        ]

    movie_list_text = "\n".join(f"- {m['title']} : {m['reason']}" for m in movies_with_reasons)
    user_message = f"Can you provide TMDb ratings for these movies?\n\nHere are movies and reasons:\n{movie_list_text}"

    try:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": user_message}],
            functions=functions,
            function_call="auto"
        )
        message = completion.choices[0].message

        if message.function_call:
            func_args = json.loads(message.function_call.arguments)
            movies = func_args.get("movies", [])

            top_movies = get_movie_ratings(movies)

            # If fewer than 3, fill with additional from original list
            if len(top_movies) < 3:
                existing_titles = {m["title"] for m in top_movies}
                for m in movies_with_reasons:
                    if m["title"] not in existing_titles:
                        top_movies.append(m)
                        existing_titles.add(m["title"])
                    if len(top_movies) == 3:
                        break

            return top_movies[:3]

        else:
            return movies_with_reasons[:3]

    except Exception as e:
        return movies_with_reasons[:3]