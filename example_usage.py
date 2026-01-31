"""
Example usage of the RAG and QCluster classes.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from rag import Encoder, RAG, QCluster


def load_environment():
    """Loading environment variables from a .env file"""
    env_path = Path(__file__).parent / ".env"

    if not env_path.exists():
        print("Warning: File .env not found!")
        print(f"Create a file {env_path} with your keys")
        return None

    load_dotenv(dotenv_path=env_path)
    return True


def get_api_key():
    """Getting an API key securely"""
    api_key = os.getenv("HF_API_KEY")

    if not api_key:
        load_environment()
        api_key = os.getenv("HF_API_KEY")

    if not api_key or api_key.startswith("your_"):
        print("Error: API key not found!")
        print("Set the HF_API_KEY environment variable or create a .env file")
        print("For more details, see README.md")
        return None

    return api_key


def load_json(file_path: str):
    """Load test data"""
    try:
        with open(file=file_path, mode="r") as f:
            documents = json.load(f)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON file")
    return documents


def main():
    print("=== Example 1: Basic RAG ===")
    api_key = get_api_key()

    # Initialize encoder
    encoder = Encoder()

    # Initialize RAG
    rag = RAG(encoder, openai_api_key=api_key)

    # Sample documents
    documents = load_json("./data/test_documents.json")

    # Fit the model
    rag.fit(documents)

    # Query the model
    query = "What cashback does the Gold credit card provide?"
    response = rag.run(query, retrieval_limit=2)
    print(f"Query: {query}")
    print(f"Response: {response}\n")

    print("=== Example 2: Question Clustering ===")
    questions = load_json("./data/test_questions.json")

    # Create QCluster instance
    qcluster = QCluster(questions_idx=list(range(len(questions))), questions=questions)

    # Cluster questions
    clusters = qcluster.cluster(n_clusters=2, show_results=True)
    print(f"\nClusters formed: {len(clusters)}")


if __name__ == "__main__":
    main()
