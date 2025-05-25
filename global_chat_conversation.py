from typing import List, Dict, Any, Union
import json

from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import HumanMessage, AIMessage
from utils import get_api_key

OPENAI_API_KEY = get_api_key("OPENAI_API_KEY")

def get_movie_chat_response(
    history: List[Dict[str, str]],
    movie_description: str,
    question: str,
    model_name: str = "gpt-4o",
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Generate a movie-related chat response using LangChain and OpenAI chat model.
    """

    def convert_messages(raw_msgs: List[Dict[str, str]]) -> List[Union[HumanMessage, AIMessage]]:
        msgs = []
        for msg in raw_msgs:
            if msg["role"] == "user":
                msgs.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                msgs.append(AIMessage(content=msg["content"]))
        return msgs

    lc_history = convert_messages(history)

    system_message_template = SystemMessagePromptTemplate.from_template(
        """
        You are a knowledgeable and helpful assistant specialized **exclusively** in answering questions about movies.

        Below you will find detailed descriptions of movies recommended to the user:

        Detailed descriptions:
        {movie_description}

        When responding, prioritize information from this context. Use your general movie knowledge only when necessary.

        You will also have access to the conversation history. In the history, look for the movie name under the section titled:
        "🎬 Here's a movie you might enjoy:"
        This indicates the movie currently presented to the user. If the user refers to "this movie," it means the one
        in that section. However, the user may also ask about other movies from the list, so be prepared to address those as well.

        Additional guidelines:
        - Answer **only** questions related to the movies or movie-related topics.
        - If the user asks about subjects outside of movies, politely inform them that your expertise is 
         limited to movies, and kindly encourage them to ask movie-related questions.
        - Aim to bring the conversation to a natural close after providing sufficient assistance; avoid endless back-and-forth.
        - Every few messages, gently ask a question like: "Would you like to continue chatting about movies, 
        or should we wrap up?" Feel free to use similar phrasing.
        - If the user indicates they want to end the conversation (e.g., says bye, thanks, no more questions), 
        call the function 'end_conversation' to signal that the chat should end gracefully.
        - Otherwise, continue to answer questions about movies.

        """
    )

    prompt = ChatPromptTemplate.from_messages([
        system_message_template,
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    formatted_prompt = prompt.format_prompt(
        movie_description=movie_description,
        history=lc_history,
        input=question
    )

    functions = [
        {
            "name": "end_conversation",
            "description": "Signal to end the conversation gracefully",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Final message to the user to wrap up the conversation"
                    }
                },
                "required": [],
            },
        }
    ]

    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        openai_api_key=OPENAI_API_KEY,
    )

    try:
        response = llm(
            formatted_prompt.to_messages(),
            functions=functions,
            function_call="auto"
        )

        if not response or (not response.content and not response.additional_kwargs.get("function_call")):
            return {
                "error": True,
                "message": "Something went wrong while generating a response. Please try again."
            }

        if response.additional_kwargs.get("function_call", {}).get("name") == "end_conversation":
            args_json = response.additional_kwargs["function_call"].get("arguments", "{}")
            try:
                args = json.loads(args_json)
                farewell = args.get("message", "Alright then! If you have more questions in the future, feel free to reach out.")
            except json.JSONDecodeError:
                farewell = "Alright then! If you have more questions in the future, feel free to reach out."

            return {
                "end_conversation": True,
                "message": farewell
            }

        return {
            "end_conversation": False,
            "message": response.content
        }

    except Exception as e:
        return {
            "error": True,
            "message": "Something went wrong while contacting the model. Please try again."
        }