import requests
from typing import List, Dict, Optional, Any

def get_movie_details(
    title: str, 
    tmdb_api_key: str, 
    max_entries: int = 3
) -> Optional[Dict[str, Any]]:
    """
    Fetch detailed information about a movie from TMDb, including cast, crew, and reviews.
    Returns None if any critical fetch fails or no results are found.
    """
    # Search for the movie by title
    search_url = "https://api.themoviedb.org/3/search/movie"
    search_params = {
        "api_key": tmdb_api_key,
        "query": title,
        "include_adult": "false",
    }
    search_resp = requests.get(search_url, params=search_params)
    if search_resp.status_code != 200:
        return None
    
    search_data = search_resp.json()
    if not search_data.get("results"):
        return None
    
    movie_id = search_data["results"][0]["id"]
    
    # Get movie details
    details_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    details_params = {
        "api_key": tmdb_api_key,
        "language": "en-US"
    }
    details_resp = requests.get(details_url, params=details_params)
    if details_resp.status_code != 200:
        return None
    
    details = details_resp.json()
    
    # Get credits (cast and crew)
    cast, crew = [], []
    credits_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits"
    credits_resp = requests.get(credits_url, params={"api_key": tmdb_api_key})
    if credits_resp.status_code == 200:
        credits = credits_resp.json()
        cast = [member["name"] for member in credits.get("cast", [])[:max_entries]]
        crew = credits.get("crew", [])[:max_entries]
    
    # Get reviews (limit to max_entries), exclude author
    reviews = []
    reviews_url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews"
    reviews_resp = requests.get(reviews_url, params={"api_key": tmdb_api_key})
    if reviews_resp.status_code == 200:
        reviews_data = reviews_resp.json()
        reviews = reviews_data.get("results", [])[:max_entries]
        reviews = [r["content"] for r in reviews]
    
    # Extract production companies and countries
    production_companies = [company["name"] for company in details.get("production_companies", [])] or []
    production_countries = [country["name"] for country in details.get("production_countries", [])] or []
    
    movie_info = {
        "title": title,
        "overview": details.get("overview", ""),
        "release_date": details.get("release_date", ""),
        "runtime": details.get("runtime", 0),
        "genres": [genre["name"] for genre in details.get("genres", [])],
        "rating": details.get("vote_average", 0),
        "cast": cast,
        "crew": [{"name": c.get("name"), "job": c.get("job")} for c in crew],
        "reviews": reviews,
        "production_companies": production_companies,
        "production_countries": production_countries,
    }
    
    return movie_info

def get_descriptions(
    recommendations: List[Dict[str, Any]], 
    tmdb_api_key: str, 
    max_entries: int = 3
) -> List[Dict[str, Any]]:
    """
    Given a list of movie recommendations, fetch detailed descriptions for each.
    If fetching details fails, returns a fallback dictionary with empty fields.
    """
    descriptions = []
    for rec in recommendations:
        title = rec.get("title")
        details = get_movie_details(title, tmdb_api_key, max_entries=max_entries)
        if details is None:
            details = {
                "title": title,
                "overview": "",
                "release_date": "",
                "runtime": 0,
                "genres": [],
                "rating": 0,
                "cast": [],
                "crew": [],
                "reviews": [],
                "production_companies": [],
                "production_countries": [],
            }
        descriptions.append(details)
    return descriptions