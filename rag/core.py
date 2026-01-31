import torch
from typing import List, Tuple, Dict
from sentence_transformers.util import pytorch_cos_sim
from openai import OpenAI
from sklearn.cluster import KMeans


class RAG:
    """
    Retrieval-Augmented Generation (RAG) class for generating responses based on retrieved documents.
    """

    def __init__(
        self,
        encoder,
        openai_api_key: str,
        model_name: str = "openai/gpt-oss-20b:groq",
    ):
        """
        Initializes the RAG class with the given encoder.

        Args:
            encoder (Encoder): The encoder to be used for encoding documents and queries.
            openai_api_key (str): API key for authenticating with the Hugging Face Inference Router.
            model_name (str): Name of the model to use for generation. Default is "openai/gpt-oss-20b:groq".

        Raises:
            ValueError: If the encoder is not an instance of Encoder, or if api_key/model_name are invalid.
        """
        from .encoder import Encoder

        if not isinstance(encoder, Encoder):
            raise ValueError("The encoder must be an instance of Encoder.")

        if not openai_api_key or not isinstance(openai_api_key, str):
            raise ValueError("openai_api_key must be a non-empty string.")

        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be a non-empty string.")

        self.documents: List[str] = []
        self.doc_embeddings: torch.Tensor = None
        self.encoder = encoder

        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=openai_api_key,
        )
        self.model_name = model_name

    def fit(self, documents: List[str]):
        """
        Fits the RAG model and calculates embeddings for the provided documents.

        Args:
            documents (List[str]): List of documents to be used for retrieval.

        Raises:
            ValueError: If the documents list is empty.
            RuntimeError: If there is an error encoding the documents.
        """
        if not documents:
            raise ValueError("The documents list cannot be empty.")

        try:
            self.documents = documents
            self.doc_embeddings = self.encoder.encode(documents)
        except Exception as e:
            raise RuntimeError(f"Failed to encode documents: {str(e)}")

    def retrieve(
        self, query: str, retrieval_limit: int = 5, similarity_threshold: float = 0.5
    ) -> Tuple[List[int], List[str]]:
        """
        Retrieves the most relevant documents based on the query.

        Args:
            query (str): The query text.
            retrieval_limit (int): Maximum number of documents to retrieve. Default is 5.
            similarity_threshold (float): Threshold for document similarity to be considered relevant. Default is 0.5.

        Returns:
            Tuple[List[int], List[str]]: The indices of the retrieved documents and the retrieved documents themselves.

        Raises:
            ValueError: If the documents have not been fitted yet.
            ValueError: If the retrieval limit is not between 1 and 10.
            ValueError: If the retrieval limit is greater than the number of documents.
            ValueError: If the similarity threshold is not between 0 and 1.
        """
        if self.documents is None or self.doc_embeddings is None:
            raise ValueError("The RAG model has not been fitted with documents.")

        if not (1 <= retrieval_limit <= 10):
            raise ValueError("Retrieval limit must be between 1 and 10.")

        if retrieval_limit > len(self.documents):
            raise ValueError("Retrieval limit cannot exceed the number of documents.")

        if not (0 <= similarity_threshold <= 1):
            raise ValueError("Similarity threshold must be between 0 and 1.")

        # Encode the query
        query_embedding = self.encoder.encode(query)

        # Compute cosine similarity
        similarities = pytorch_cos_sim(query_embedding, self.doc_embeddings)[0]

        # Get top-k documents
        top_k = torch.topk(similarities, retrieval_limit)

        indices = top_k.indices.tolist()
        scores = top_k.values.tolist()

        # Filter by similarity threshold
        filtered = [
            (i, s) for i, s in zip(indices, scores) if s >= similarity_threshold
        ]

        if not filtered:
            return [], []

        final_indices = [i for i, _ in filtered]
        retrieved_docs = [self.documents[i] for i in final_indices]

        return final_indices, retrieved_docs

    def _create_prompt_template(self, query: str, retrieved_docs: List[str]) -> str:
        """
        Creates a prompt template for text generation.

        Args:
            query (str): The user query.
            retrieved_docs (List[str]): The list of retrieved documents.

        Returns:
            str: The formatted prompt.
        """
        prompt = "Instructions: Based on the relevant documents, generate a comprehensive response to the user's query.\n"

        prompt += "Relevant Documents:\n"
        for i, doc in enumerate(retrieved_docs):
            prompt += f"Document {i+1}: {doc}\n"

        prompt += f"User Query: {query}\n"

        return prompt

    def _generate(self, query: str, retrieved_docs: List[str]) -> str:
        """
        Generates a response based on the retrieved documents and query.

        Args:
            query (str): The user query.
            retrieved_docs (List[str]): The list of retrieved documents.

        Returns:
            str: The generated response.

        Pseudo-code:
            - Create a prompt using the query and retrieved documents.
            - Pass the prompt to a text generation model.
            - Retrieve and return the generated response.
        """
        if not retrieved_docs:
            return "Sorry, I couldn't find any suitable documents to answer."

        # Create the prompt template
        prompt = self._create_prompt_template(query, retrieved_docs)

        # Pass the prompt to the text generation model
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are Kimi, an AI assistant created by Moonshot AI.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    },
                ],
            )
        except Exception as e:
            return f"Error generating response via OpenAI API: {str(e)}"
        generated_response = completion.choices[0].message.content
        return generated_response

    def run(
        self, query: str, retrieval_limit: int = 1, similarity_threshold: float = 0.5
    ) -> str:
        """
        Runs the full RAG pipeline: retrieves documents and generates a response.

        Args:
            query (str): The user query.

        Returns:
            str: The generated response.
        """
        _, retrieved_docs = self.retrieve(query, retrieval_limit, similarity_threshold)
        generated_response = self._generate(query, retrieved_docs)

        return generated_response


class QCluster:
    """A class used to cluster questions using k-means clustering."""

    def __init__(self, questions_idx: List[int], questions: List[str]):
        """
        Initializes the QCluster with question indices and questions.

        Args:
            questions_idx (List[int]): Indices of the questions.
            questions (List[str]): List of questions to be clustered.

        Raises: ValueError: If questions_idx and questions lists have different lengths or are empty.
        """
        if not questions_idx or not questions:
            raise ValueError("Question indices and questions lists cannot be empty.")
        if len(questions_idx) != len(questions):
            raise ValueError("Length of questions_idx must equal length of questions.")

        self.questions_idx = questions_idx
        self.questions = questions
        from .encoder import Encoder

        self.encoder = Encoder()
        self.embeddings = None
        self.cluster_labels = None
        self.clusters = None

    def cluster(
        self, n_clusters: int, show_results: bool = False
    ) -> Dict[int, List[int]]:
        """
        Clusters the questions into the specified number of clusters using k-means clustering.

        Args:
            n_clusters (int): The number of clusters to form. Must be between 1 and 10.
            show_results (bool, optional): Whether to print the clusters (default is False).

        Returns:
            Dict[int, List[int]]: A dictionary where the keys are cluster labels and the values are lists of question indices in each cluster.

        Raises:
            ValueError: If n_clusters is not between 1 and 10.

        Notes:
            The k-means algorithm is initialized with random_state=42 to ensure reproducibility of the results.
        """
        if not isinstance(n_clusters, int) or n_clusters < 1 or n_clusters > 10:
            raise ValueError("n_clusters must be an integer between 1 and 10.")

        self.embeddings = self.encoder.encode(self.questions).cpu().numpy()

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = kmeans.fit_predict(self.embeddings)

        self.clusters = {}
        for label, idx in zip(self.cluster_labels, self.questions_idx):
            if label not in self.clusters:
                self.clusters[int(label)] = []
            self.clusters[int(label)].append(idx)

        if show_results:
            self.print_clusters()

        return self.clusters

    def print_clusters(self):
        """Prints the clusters with their respective question indices and questions."""
        if self.clusters is None:
            print("No clusters available. Call .cluster() first.")
            return

        print("=" * 60)
        print("CLUSTERS OF QUESTIONS")
        print("=" * 60)
        for label, indices in sorted(self.clusters.items()):
            print(f"\nCluster {label}:")
            for idx in indices:
                question_idx = self.questions_idx.index(idx)
                question_text = self.questions[question_idx]
                print(f'  - Question index: {idx}, Text: "{question_text}"')
        print("=" * 60)
