from vector_search import VectorSearchEngine
import numpy as np

docs = [
    "Le machine learning est un sous-domaine de l’IA",
    "Les réseaux de neurones sont très utilisés",
    "HNSW est une méthode de recherche approximate nearest neighbor",
    "Python est un langage flexible",
    "Les embeddings vectorisent le texte"
]

engine = VectorSearchEngine(dimension=128)
engine.load_or_create_index(docs)

# Simule une requête avec un vecteur aléatoire
query = np.random.rand(1, 128).astype(np.float32)

results = engine.search(query, k=3)

for res in results:
    print(f"- ID: {res['id']} | Distance: {res['distance']:.3f} | Document: {res['document']}")
