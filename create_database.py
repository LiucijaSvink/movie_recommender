from typing import Callable
import pandas as pd
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from utils import get_api_key
import argparse

OPENAI_API_KEY = get_api_key("OPENAI_API_KEY")
QDRANT_URL = "https://4f78837f-a98f-4bca-b598-903c86199ef2.eu-west-2-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = get_api_key("QDRANT_API_KEY")
COLLECTION_NAME = "movies_cluster"

def row_to_document(row: pd.Series) -> Document:
    """
    Convert a movie DataFrame row into a LangChain Document.
    """
    text_chunks = [
        f"Movie title: {row['title']}",
        f"Overview: {row['overview']}",
        f"Genres: {', '.join(row['genres'])}" if row['genres'] else "",
        f"Cast: {', '.join(row['cast'])}" if row['cast'] else "",
    ]
    full_text = "\n".join(chunk for chunk in text_chunks if chunk)
    return Document(page_content=full_text)

def create_qdrant_movie_db(
    movie_db_path: str,
    openai_api_key: str = OPENAI_API_KEY,
    qdrant_url: str = QDRANT_URL,
    qdrant_api_key: str = QDRANT_API_KEY,
    collection_name: str = COLLECTION_NAME,
    row_to_doc_fn: Callable[[pd.Series], Document] = row_to_document
) -> Qdrant:
    """
    Create a Qdrant vector store from a movie database parquet file using OpenAI embeddings.
    """
    movie_database = pd.read_parquet(movie_db_path)
    documents = [row_to_doc_fn(row) for _, row in movie_database.iterrows()]

    embedding = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)
    embedding_dimensions = len(embedding.embed_query("test"))

    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dimensions, distance=Distance.COSINE)
    )

    vectorstore = Qdrant.from_documents(
        documents=documents,
        embedding=embedding,
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=collection_name
    )

    return vectorstore

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Qdrant movie database from a parquet file.")
    parser.add_argument("movie_db_path", type=str, help="Path to the movie parquet file")
    args = parser.parse_args()

    print(f"Creating Qdrant movie database from {args.movie_db_path} ...")
    vectorstore = create_qdrant_movie_db(args.movie_db_path)
    print("Qdrant movie database created successfully.")