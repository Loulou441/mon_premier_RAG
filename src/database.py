import os
import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME, DB_PATH, COLLECTION_NAME

class VectorDB:
    def __init__(self, chunks=None):
        """Aiguillage automatique : charge ou crée la base persistante."""
        self.client = chromadb.PersistentClient(path=DB_PATH)
        collection_name = COLLECTION_NAME
        
        # Création de la base
        if not chunks:
            raise ValueError("La base de données n'existe pas et aucun chunk n'a été fourni pour la créer.")
        
        print(f"Initialisation et création de la base avec le modèle : {EMBEDDING_MODEL_NAME}")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # Sauvegarde du nom du modèle dans la métadonnée de collection 
        # Astuce anti-bug : empêche d'interroger une DB existante avec un modèle d'embedding différent 
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"embedding_model": EMBEDDING_MODEL_NAME}
        )
        self.initiation_database(chunks)

    def initiation_database(self, chunks):
        """Encode et insère les éléments dans ChromaDB."""
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [{"id": chunk["id"], "source": chunk["source"], "category": chunk["category"]} for chunk in chunks]
        
        # Encodage avec normalisation
        embeddings = self.model.encode(
            texts, 
            batch_size=32, 
            normalize_embeddings=True, 
            show_progress_bar=True
        ).tolist()
        
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print("Ingestion terminée avec succès.")