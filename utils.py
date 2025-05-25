import streamlit as st
from langchain.schema import Document
import pycountry
import re
from typing import Dict, Any, List, Optional

def get_api_key(key_name: str = "OPEN_API_KEY") -> str:
    """Retrieve an API key from Streamlit secrets."""
    api_key = st.secrets[key_name]
    return api_key

def row_to_document(row: Dict[str, Any]) -> Document:
    """Convert a dictionary row of movie data into a LangChain Document object."""
    text_chunks = [
        f"Movie title: {row['title']}",
        f"Overview: {row['overview']}",
        f"Genres: {row['genres']}" if row['genres'] else "",
        f"Cast: {row['cast']}" if row['cast'] else "",
    ]
    full_text = "\n".join([chunk for chunk in text_chunks if chunk])
    document = Document(page_content=full_text, metadata={})
    return document

def get_countries() -> List[str]:
    """Return a sorted list of all country names."""
    countries = [country.name for country in pycountry.countries]
    countries_sorted = sorted(countries)
    return countries_sorted

def get_country_code(country_name: str) -> Optional[str]:
    """Return the ISO alpha-2 country code for a given country name."""
    try:
        country = pycountry.countries.lookup(country_name.strip())
        country_code = country.alpha_2
    except LookupError:
        country_code = None
    return country_code

def clean_input_text(input_text: str) -> str:
    """
    Clean the input text by:
    1. Removing any non-alphabetic characters except spaces, apostrophes, and hyphens.
    2. Converting the text to lowercase.
    3. Checking if the input is valid (not only spaces, apostrophes, or hyphens).
    4. Ensuring the input is within the max character limit of 150 characters.
    """
    cleaned_input = re.sub(r'[^a-zA-Z\s\'-]', '', input_text)
    cleaned_input = cleaned_input.lower()

    # Check for empty input or invalid characters
    if cleaned_input == "" or re.match(r'^[\s\'-]*$', cleaned_input):
        return "Invalid input. Your input can only include letters, spaces, apostrophes, and hyphens."
    
    # Check if the input exceeds 150 characters
    if len(cleaned_input) > 150:
        return "The input exceeds the maximum length of 150 characters. Please provide a shorter job title."

    return cleaned_input