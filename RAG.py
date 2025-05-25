from typing import List
from pydantic import BaseModel
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from qdrant_client import QdrantClient
from utils import get_api_key
from langsmith import traceable

import os

os.environ["LANGSMITH_API_KEY"] = get_api_key("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "movie-recommender"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

class MovieRecommendation(BaseModel):
    title: str
    reason: str

class RecommendationList(BaseModel):
    recommendations: List[MovieRecommendation]

OPENAI_API_KEY = get_api_key("OPENAI_API_KEY")
QDRANT_API_KEY = get_api_key("QDRANT_API_KEY")
URL = "https://4f78837f-a98f-4bca-b598-903c86199ef2.eu-west-2-0.aws.cloud.qdrant.io"

@traceable(name="get_movie_recommendations")
def get_movie_recommendations(themes: str, genres: str, actors: str) -> List[MovieRecommendation]:
    """
    Generate movie recommendations based on user preferences.
    """
    embedding: OpenAIEmbeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=OPENAI_API_KEY,
    )

    client: QdrantClient = QdrantClient(
        url=URL,
        api_key=QDRANT_API_KEY
    )

    vectorstore: Qdrant = Qdrant(
        client=client,
        collection_name="movies_cluster",
        embeddings=embedding,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    inputs: List[str] = [themes, genres, actors]
    all_retrieved_docs = []

    for input_text in inputs:
        docs = retriever.get_relevant_documents(input_text)
        all_retrieved_docs.extend(docs)

    unique_docs = list({doc.page_content: doc for doc in all_retrieved_docs}.values())
    retrieved_docs: str = "\n**\n".join(doc.page_content for doc in unique_docs)

    parser = PydanticOutputParser(pydantic_object=RecommendationList)
    format_instructions = parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful AI movie assistant. I am seeking for movie recommendations.

        Here are my preferences:
        - Topics: {topics}
        - Genres: {genres}
        - Favorite Actors: {actors}

        Here are some movie descriptions retrieved based on these preferences:

        {retrieved_docs}

        Recommend exactly 9 movies that I am likely to enjoy. For each recommended movie, provide a concise 
        explanation why is recommended - include one sentence referring to it's overview and one sentence explaning why is it relevant for me.
        Return the result as a structured JSON with a 'recommendations' list, where each item has 'title' and 'reason'.
        The output JSON should be valid and parsable.
        """
    ).partial(format_instructions=format_instructions)

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
        api_key=OPENAI_API_KEY,
    )

    chain = prompt | llm | parser

    response = chain.invoke({
        "topics": themes,
        "genres": genres,
        "actors": actors,
        "retrieved_docs": retrieved_docs,
    })

    return response.recommendations
