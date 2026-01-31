"""
Example usage of the RAG and QCluster classes.
"""

import os
from rag import Encoder, RAG, QCluster
from dotenv import load_dotenv

load_dotenv(".env")


def main():
    # Example 1: Basic RAG usage
    print("=== Example 1: Basic RAG ===")

    # Initialize encoder
    encoder = Encoder()

    # Initialize RAG
    api_key = os.getenv("HF_API_KEY", "your-huggingface-api-key-here")
    rag = RAG(encoder, openai_api_key=api_key)

    # Sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Python is a popular programming language for data science.",
        "Transformers are a type of neural network architecture.",
    ]

    # Fit the model
    rag.fit(documents)

    # Query the model
    query = "What is machine learning?"
    response = rag.run(query, retrieval_limit=2)
    print(f"Query: {query}")
    print(f"Response: {response}\n")

    # Example 2: Question clustering
    print("=== Example 2: Question Clustering ===")

    questions = [
        "What is machine learning?",
        "How does deep learning work?",
        "What are neural networks?",
        "What is Python used for?",
        "How to learn programming?",
        "What is data science?",
    ]

    # Create QCluster instance
    qcluster = QCluster(questions_idx=list(range(len(questions))), questions=questions)

    # Cluster questions
    clusters = qcluster.cluster(n_clusters=2, show_results=True)

    print(f"\nClusters formed: {len(clusters)}")


if __name__ == "__main__":
    main()
