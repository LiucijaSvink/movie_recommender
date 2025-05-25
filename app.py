import streamlit as st
from utils import get_api_key, get_countries, clean_input_text
from typing import Dict

from RAG import get_movie_recommendations
from movie_ratings import run_movie_rating_search
from movie_trailer_search import run_movie_trailer_search
from movie_stream_search import run_streaming_search
from global_chat_conversation import get_movie_chat_response
from movie_descriptions import get_descriptions
from validation import validate_input

OPENAI_API_KEY = get_api_key("OPENAI_API_KEY")
TMDB_API_KEY = get_api_key("TMDB_API_KEY")
ASSISTANT_INTRO = "Feel free to ask me about the recommended movies (e.g., ratings, reviews, actors) or movies in general."


initial_prompt = (
        "Hi! I'm an AI-powered movie recommendation assistant, optimized for discovering the latest films.\n"
        "My goal is to help you find the perfect pick for your next movie night. 🍿\n\n"
        "To understand your preferences better, I'd love to ask you a few quick questions.\n\n"
    )

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'conversation_started' not in st.session_state:
        st.session_state.conversation_started = False
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {
            "themes": None,
            "genres": None,
            "actors": None
        }
    if 'recommendations_generated' not in st.session_state:
        st.session_state.recommendations_generated = False
    if 'current_recommendation_index' not in st.session_state:
        st.session_state.current_recommendation_index = 0
    if 'all_recommendations' not in st.session_state:
        st.session_state.all_recommendations = []

def format_recommendation_text(movie: Dict) -> str:
    """Format the recommendation message for a single movie"""
    return (
        f"🎬 Here's a movie you might enjoy:\n\n"
        f"**{movie['title']}**\n\n"
        f"{movie['reason']}\n\n"
        f"What would you like to do next?"
    )

def get_question(question_number: int) -> str:
    """Get the question based on the current question number"""
    questions = [
        "What particular topics, themes, or storylines are you interested in? (e.g., self-discovery, space exploration, coming of age, historical events)",
        "What genres do you typically enjoy? (e.g., comedy, action, drama, sci-fi)",
        "Which actors or actresses do you like? (e.g., Margot Robbie, Florence Pugh, Ryan Gosling, Timothée Chalamet)"
    ]
    return questions[question_number] if question_number < len(questions) else None

def process_user_input(user_input: str) -> str:
    """Process user input and generate appropriate response"""
    current_question = st.session_state.current_question

    if current_question == 0:
        st.session_state.user_preferences["themes"] = user_input
        st.session_state.current_question = 1
        question = get_question(1)
        return question

    elif current_question == 1:
        st.session_state.user_preferences["genres"] = user_input
        st.session_state.current_question = 2
        question = get_question(2)
        return question

    elif current_question == 2:
        st.session_state.user_preferences["actors"] = user_input
        st.session_state.recommendations_generated = False
        recommendations_text = generate_recommendation()
        recommendations = st.session_state.all_recommendations
        st.session_state.movie_descriptions = get_descriptions(recommendations, TMDB_API_KEY, max_entries=3)

        return recommendations_text

def show_recommendation_actions():
    """Display action buttons for the current recommendation"""
    if st.session_state.get('continue_chat', False):
        return

    total = len(st.session_state.all_recommendations)
    current_index = st.session_state.current_recommendation_index
    is_last = current_index >= total - 1
    current_movie = st.session_state.all_recommendations[current_index]

    countries = get_countries()

    with st.container():

        # Suggest another movie
        if not is_last:
            if st.button("🎞️ Suggest another movie", key=f"suggest_another_{current_index}"):
                st.session_state.current_recommendation_index += 1

                if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
                    st.session_state.messages.pop()

                current_movie = st.session_state.all_recommendations[st.session_state.current_recommendation_index]
                recommendation_text = format_recommendation_text(current_movie)

                st.session_state.messages.append({"role": "assistant", "content": recommendation_text})
                st.rerun()
        
        # Movie trailer search
        trailer_key = f"trailer_url_{current_index}"

        if st.button("Search for movie trailer", key=f"trailer_button_{current_index}"):
            title = st.session_state.all_recommendations[current_index]['title']
            with st.spinner(f"Searching trailer for **{title}**..."):
                trailer_url = run_movie_trailer_search(title)

            if trailer_url:
                st.session_state[trailer_key] = trailer_url
            else:
                st.session_state[trailer_key] = None 
                
        # Show trailer if it exists
        if trailer_key in st.session_state and st.session_state[trailer_key]:
            st.video(st.session_state[trailer_key])
        elif trailer_key in st.session_state and st.session_state[trailer_key] is None:
            st.warning(f"No trailer found for **{current_movie['title']}**.")
        
        # Streaming provider with country dropdown
        country_select_key = f"show_country_select_{current_index}"
        if country_select_key not in st.session_state:
            st.session_state[country_select_key] = False

        if st.button("Search for streaming provider", key=f"provider_button_{current_index}"):
            st.session_state[country_select_key] = True
            st.session_state[f"country_select_{current_index}"] = "Select a country..."
            st.session_state.pop(f"streaming_result_{current_index}", None)
            st.session_state['last_country'] = None

        if st.session_state[country_select_key]:
            countries_with_placeholder = ["Select a country..."] + countries

            selected_country = st.selectbox(
                "Select your country:",
                countries_with_placeholder,
                index=0,  # placeholder selected by default
                key=f"country_select_{current_index}"
            )

            result_key = f"streaming_result_{current_index}"

            if selected_country != "Select a country...":
                if "last_country" not in st.session_state or st.session_state.last_country != selected_country:
                    st.session_state.last_country = selected_country
                    with st.spinner(f"Searching for streaming providers in {selected_country}..."):
                        result = run_streaming_search(current_movie['title'], selected_country)
                    st.session_state[result_key] = result or ""

            result = st.session_state.get(result_key)
            if result:
                st.write(result)
            elif result == "":
                st.warning(f"No streaming provider found for **{current_movie['title']}** in {selected_country}.")

        # Continue chat
        if st.button("Continue conversation", key="continue_chat_btn"):
            st.session_state.continue_chat = True

            user_msg = "Continue conversation"
            st.session_state.messages.append({
                    "role": "user",
                    "content": user_msg
                })
            
            # Only add the assistant message once
            if not st.session_state.messages or st.session_state.messages[-1]["content"] != ASSISTANT_INTRO:
                new_message = {
                    "role": "assistant",
                    "content": ASSISTANT_INTRO
                }
                st.session_state.messages.append(new_message)
                with st.chat_message("assistant"):
                    st.write(new_message["content"])
            
            st.rerun()

def generate_recommendation() -> str:
    """Generate movie recommendation based on user preferences"""
    if st.session_state.recommendations_generated:
        return None

    st.session_state.recommendations_generated = True
    preferences = st.session_state.user_preferences

    with st.spinner("🎬 Generating recommendations..."):
        try:
            recommendations = get_movie_recommendations(
                themes=preferences['themes'],
                genres=preferences['genres'],
                actors=preferences['actors']
            )

            if not recommendations:
                return "Sorry, I couldn't find any recommendations based on your preferences."

            top_movies = run_movie_rating_search(recommendations)
            st.session_state.all_recommendations = top_movies

            current_movie = top_movies[0]
            recommendation_text = format_recommendation_text(current_movie)

            return recommendation_text

        except Exception as e:
            st.error(f"An error occurred while generating recommendations: {str(e)}")
            return "I apologize, but I encountered an error while generating recommendations. Please try again."

def main():

    st.set_page_config(page_title="🎬 Movie Recommender Chatbot")
    st.title("🎥 AI Movie Recommendation Assistant")

    initialize_session_state()

    if not st.session_state.conversation_started:
        st.write(initial_prompt)
        if st.button("Let's Get Started! 🎬"):
            st.session_state.conversation_started = True
            first_question = get_question(0)
            st.session_state.messages.append({"role": "assistant", "content": first_question})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.conversation_started:
        if not st.session_state.recommendations_generated:
            question = get_question(st.session_state.current_question)
            if prompt := st.chat_input("Your answer..."):
                cleaned_input = clean_input_text(prompt)
                validation_result = validate_input(cleaned_input)

                if "Invalid input" in cleaned_input or "too long" in cleaned_input:
                    st.warning(cleaned_input)

                else:
                    if validation_result == "no":
                        st.warning("The input was not recognized as a valid or specific " \
                        "enough for describing movies or actors. This may lead to unexpected results. " \
                        "For better results, please start over with a concise keywords or phrases.")

                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.write(prompt)

                    response = process_user_input(prompt)

                    if response:
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        with st.chat_message("assistant"):
                            st.write(response)

                        if st.session_state.recommendations_generated:
                            show_recommendation_actions()
        else:
            show_recommendation_actions()
            
            if st.session_state.get('continue_chat', False):

                if st.session_state.get('chat_ended', False):
                    
                    # Conversation ended: show farewell message and disabled input + start over button
                    farewell_message = st.session_state.get('farewell_message', "Thanks for chatting! Goodbye.")
                    st.markdown(f"{farewell_message}")
                    
                    st.text_input("Conversation ended. Please start over.", disabled=True)

                else:
                    prompt = st.chat_input("Ask me about movies...")
                    if prompt:
                        cleaned_input = clean_input_text(prompt)
                        if "Invalid input" in cleaned_input or "too long" in cleaned_input:
                            st.warning(cleaned_input)
                        else:
                            movie_descriptions = st.session_state.movie_descriptions
                            response = get_movie_chat_response(st.session_state.messages, movie_descriptions, prompt)

                            st.session_state.messages.append({"role": "user", "content": prompt})
                            with st.chat_message("user"):
                                st.write(prompt)

                            if response.get("end_conversation", False):
                                farewell_message = response.get("message", "Thanks for chatting! Goodbye.")
                                st.session_state.chat_ended = True
                                st.session_state.farewell_message = farewell_message
                            else:
                                st.session_state.messages.append({"role": "assistant", "content": response.get("message", "")})
                                with st.chat_message("assistant"):
                                    st.write(response.get("message", ""))

                            st.rerun()

        # Add Start over button on the right side
        col1, col2, col3 = st.columns([1, 2, 1])
        with col3:
            if st.button("Start over", key="start_over_btn"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
    
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center; margin-top:1rem;">
            <a href="https://www.themoviedb.org/" target="_blank" rel="noopener noreferrer">
                <img src="https://www.themoviedb.org/assets/2/v4/logos/v2/blue_square_2-d537fb228cf3ded904ef09b136fe3fec72548ebc1fea3fbbd1ad9e36364db38b.svg" 
                alt="TMDb Logo" style="height:25px;"/>
            </a>
        </div>
        <div style="font-size:12px; color:gray; text-align:center; margin-top:0.5rem;">
            This product uses the TMDb API but is not endorsed or certified by TMDb.
        </div>
        """, unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 