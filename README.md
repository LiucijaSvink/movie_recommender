# Movie Recommendation Chatbot

An AI-powered chatbot that provides personalized movie recommendations and consultation using Retrieval-Augmented Generation (RAG) with LangChain, OpenAI embeddings, and Qdrant vector search.

## Try the App
[Link to Movie Recommender](https://movierecommender-3kqm9krphhega4p5qgpzrc.streamlit.app/)

## Features

- **Personalized Recommendations**: Get movie suggestions based on your preferences and conversation history  
- **Multi-Modal Search**: Includes movie ratings, trailers, streaming availability, and detailed descriptions  
- **Conversational Interface**: Friendly chat UI built with Streamlit  
- **Fast Vector Search**: Uses Qdrant for efficient semantic search over movie data  
- **OpenAI Embeddings Support**: Enhanced recommendations powered by OpenAI embeddings  

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Set up your API keys:
   - Create a `.streamlit/secrets.toml` file
   - Add your OpenAI API key:
```toml
OPENAI_API_KEY = "your-openai-api-key-here"
TMDB_API_KEY = "your-tmdb-api-key-here"
QDRANT_API_KEY = "your-qdrant-api-key-here"
LANGSMITH_API_KEY = "your-langsmith-api-key-here"
```

## Usage

1. Run the application:
```bash
streamlit run app.py
```
Chat with the bot to get movie recommendations based on your preferences.

## Files Description

- **RAG.py**  
  Implements the Retrieval-Augmented Generation logic combining LangChain, OpenAI embeddings, and Qdrant vector search to generate movie recommendations.

- **app.py**  
  The main Streamlit application script providing the chatbot interface for movie recommendations.

- **create_database.py**  
  Script to create and populate the movie database used for recommendations. Should be run once before starting the application.

- **global_chat_conversation.py**  
  Handles global chat state management and conversation history across user interactions.

- **movie_descriptions.py**  
  Contains functions or data related to fetching, parsing, or managing detailed movie descriptions.

- **movie_ratings.py**  
  Uses TMDb to fetch ratings for a list of movies and returns the top 3 highest-rated titles via OpenAI function calling.

- **movie_stream_search.py**  
  Finds streaming platforms for a movie in a user’s country using TMDb and OpenAI function calling.

- **movie_trailer_search.py**  
  Fetches official movie trailer URLs from TMDb using OpenAI function calling. Supports YouTube and Vimeo trailers.

- **requirements.txt**  
  Lists all Python dependencies required to run the project.

- **utils.py**  
  Utility functions used across different modules for common tasks and helpers.

- **validation.py**  
  Uses OpenAI to validate whether user input is a valid movie keyword, genre, actor name, or theme. Returns `"yes"` or `"no"` using a structured response model.

## Requirements

- Python 3.8+
- streamlit==1.45.1
- openai==1.79.0
- langchain==0.3.25
- qdrant-client==1.14.2
- tiktoken 

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for providing the language model API  
- LangChain for the RAG framework  
- Qdrant for vector search technology  
- Streamlit for the web framework  
- TMDb for the movie database and API
