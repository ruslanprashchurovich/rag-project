import torch
from typing import List, Union
from sentence_transformers import SentenceTransformer


class Encoder:
    """
    Text encoder using sentence-transformers for creating embeddings.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        """
        Initialize the encoder with a pretrained model.

        Args:
            model_name (str): Name of the sentence-transformer model
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(model_name)
        self.model.to(device)
        self.device = device

    def encode(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode text(s) into embeddings.

        Args:
            texts: Single string or list of strings

        Returns:
            torch.Tensor: Embeddings tensor
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts, convert_to_tensor=True, device=self.device, normalize_embeddings=True
        )
        return embeddings

    def __call__(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Alias for encode method."""
        return self.encode(texts)
