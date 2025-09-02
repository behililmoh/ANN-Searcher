# 🔍 Vector Search Engine avec HNSWLib

Ce projet implémente un moteur de recherche vectorielle simple en Python utilisant **[hnswlib](https://github.com/nmslib/hnswlib)** pour la recherche de similarité rapide entre documents textuels.

---

## 🚀 Fonctionnalités

- Création automatique d'un index HNSW à partir de documents textuels
- Recherche des documents les plus similaires à une requête vectorielle
- Sauvegarde et chargement de l'index et des embeddings
- Prise en charge facile d'autres types d'embeddings (OpenAI, Sentence Transformers, etc.)

---

## 📦 Dépendances

```bash
pip install hnswlib numpy

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


mon_projet/
├── vector_search.py
├── recherche_test.py
├── index_data/
└── README.md



