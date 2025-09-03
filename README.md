🔍 Vector Search Engine avec HNSWLib

This project implémente a simple vector search engine in Python using hnswlib for a rapide similarity search between text documents.

🚀 Fonctionnalités

    Automatic creation of an HNSW index à partir de text documents

    Search for the most similar documents to a vector query

    Save and load the index et les embeddings

    Easy support for other types d'embeddings (OpenAI, Sentence Transformers, etc.)

    📦 Dépendances


```bash

pip install hnswlib numpy

from vector_search import VectorSearchEngine
import numpy as np

docs = [
    "Machine learning is a sub-domaine de l'IA",
    "Neural networks are très utilisés",
    "HNSW is an approximate nearest neighbor search méthode",
    "Python is a flexible language",
    "Embeddings vectorize the text"
]

engine = VectorSearchEngine(dimension=128)
engine.load_or_create_index(docs)

# Simulates a query with a random vector
query = np.random.rand(1, 128).astype(np.float32)

results = engine.search(query, k=3)

for res in results:
    print(f"- ID: {res['id']} | Distance: {res['distance']:.3f} | Document: {res['document']}")


```

mon_projet/
├── vector_search.py
├── recherche_test.py
├── index_data/
└── README.md
