# RAG Project

A simple Retrieval-Augmented Generation (RAG) implementation with question clustering capabilities.

## Features

- Document retrieval using semantic similarity
- Response generation via Hugging Face Inference Router
- Question clustering using k-means
- Easy-to-use API

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from rag import Encoder, RAG

# Initialize encoder
encoder = Encoder()

# Initialize RAG with your API key
rag = RAG(encoder, openai_api_key="your-api-key")

# Fit with documents
documents = ["Document 1 text...", "Document 2 text..."]
rag.fit(documents)

# Run query
response = rag.run("Your query here")
print(response)
```

## Configuration

Create a `.env` file:

```env
# Model settings
HF_API_KEY="your-huggingface-api-key-here"
```

## Usage Examples

See `example_usage.py` for complete examples.

## API Keys

You need:

1. Hugging Face Inference Router API key for text generation
2. (Optional) Hugging Face token for downloading models

## License

MIT
