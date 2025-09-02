import os
import numpy as np
import hnswlib

class VectorSearchEngine:
    def __init__(self, dimension=128):
        self.dimension = dimension
        self.documents = []
        self.hnsw_index = None
        self.embeddings = None
        
    def load_or_create_index(self, documents, ef_construction=200, M=16):
        """Charge un index existant ou en crée un nouveau"""
        self.documents = documents
        
        # Vérifier si l'index existe déjà
        if os.path.exists("index_data/hnsw_index.bin") and os.path.exists("index_data/vectors_diskann.dat"):
            print("Chargement des index existants...")
            self.load_index()
        else:
            print("Création de nouveaux index...")
            self.create_index(documents, ef_construction, M)
    
    def create_index(self, documents, ef_construction=200, M=16):
        """Crée de nouveaux index"""
        # Générer les embeddings factices (à remplacer par ton modèle réel)
        self.embeddings = np.random.rand(len(documents), self.dimension).astype(np.float32)
        
        # Créer l'index HNSW
        self.hnsw_index = hnswlib.Index(space="cosine", dim=self.dimension)
        self.hnsw_index.init_index(max_elements=len(documents), 
                                  ef_construction=ef_construction, M=M)
        self.hnsw_index.add_items(self.embeddings, ids=list(range(len(documents))))
        
        # Sauvegarder
        os.makedirs("index_data", exist_ok=True)
        self.hnsw_index.save_index("index_data/hnsw_index.bin")
        
        # Sauvegarder les vecteurs bruts
        fp = np.memmap("index_data/vectors_diskann.dat", 
                      dtype='float32', mode='w+', shape=self.embeddings.shape)
        fp[:] = self.embeddings[:]
        del fp
    
    def load_index(self):
        """Charge les index existants"""
        # Charger HNSW
        self.hnsw_index = hnswlib.Index(space="cosine", dim=self.dimension)
        self.hnsw_index.load_index("index_data/hnsw_index.bin")
        
        # Charger les vecteurs (pour affichage)
        self.embeddings = np.memmap("index_data/vectors_diskann.dat", 
                                   dtype='float32', mode='r')
        self.embeddings = self.embeddings.reshape(-1, self.dimension)
    
    def search(self, query_vector, k=5):
        """Recherche les k documents les plus proches"""
        if self.hnsw_index is None:
            raise ValueError("Index non chargé ou créé.")
        
        labels, distances = self.hnsw_index.knn_query(query_vector, k=k)
        
        results = []
        for idx, dist in zip(labels[0], distances[0]):
            results.append({
                "document": self.documents[idx],
                "id": int(idx),
                "distance": float(dist)
            })
        return results
