from typing import Literal
from pydantic import BaseModel
from openai import OpenAI
from utils import get_api_key

class ValidationOutput(BaseModel):
    """Schema for the structured validation result returned by the model."""
    input_value: str
    validation_result: Literal["yes", "no"]

def validate_input(input_value: str) -> str:
    """Validate whether the input is a relevant and concise movie keyword, genre, actor name, or theme."""
    
    system_prompt = "You are an expert in movies, genres, keywords, and actors."
    
    user_prompt = f"""
    Your task is to verify if the input is a valid movie keyword, genre, actor name, or theme/topic/storyline.

    - It should be a short phrase describing either a genre, actor, or movie theme/storyline.
    - If the text contains more than just keywords, return "no". For example, "I am a big fan of love stories"
    is invalid because it does not represent a keyword but a full sentence.
    - Examples of valid genres: "comedy", "thriller", "sci-fi"
    - Examples of valid actors: "Tom Hanks", "Natalie Portman"
    - Examples of valid themes or topics: "losing a loved one", "revenge", "coming of age", "friendship", "family issues"
    - You should be stricter with genres and actors, but can be a bit more liberal with topics/themes.

    Return "yes" if it fits one of these categories and is relevant and concise. Otherwise, return "no".

    Input: "{input_value}"
    """

    OPENAI_API_KEY: str = get_api_key("OPENAI_API_KEY")
    client = OpenAI(api_key=OPENAI_API_KEY)

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=messages,
            temperature=0,
            response_format=ValidationOutput
        )

        parsed_content = response.choices[0].message.parsed
        validation_result: str = parsed_content.validation_result
        
        return validation_result

    except Exception as e:
        return "error"