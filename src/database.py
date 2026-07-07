import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME, DB_PATH, COLLECTION_NAME

class VectorDB:
    def __init__(self, chunks=None):
        """Aiguillage automatique : charge ou crée la base persistante."""
        self.client = chromadb.PersistentClient(path=DB_PATH)
        collection_name = COLLECTION_NAME
        existing_collections = [col.name for col in self.client.list_collections()]

        if collection_name in existing_collections:
            # Rechargement de la base
            self.collection = self.client.get_collection(name=collection_name)
            # Récupération du modèle dynamique depuis la métadonnée
            metadata = self.collection.metadata
            model_name = metadata.get("embedding_model", EMBEDDING_MODEL_NAME)
            print(f"Base détectée. Chargement du modèle depuis métadonnées: {model_name}")
            self.model = SentenceTransformer(model_name)
        else:
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
        metadatas = [{"id": chunk["id"], "source": chunk["source"], "categorie": chunk["categorie"]} for chunk in chunks]
        ids = [chunk["id"] for chunk in chunks]

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
            metadatas=metadatas,
            ids=ids
        )
        print("Ingestion terminée avec succès.")
    
    def retrieve(self, question: str, n: int = 5) -> list:
        """Encode la question et extrait les n chunks les plus proches."""
        query_vector = self.model.encode(question, normalize_embeddings=True).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=n
        )
        
        # Restructuration propre des résultats
        formatted_results = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    "text": results['documents'][0][i],
                    "id": results['metadatas'][0][i]['id'] if results['metadatas'] else "Inconnue",
                    "source": results['metadatas'][0][i]['source'] if results['metadatas'] else "Inconnue",
                    "categorie": results['metadatas'][0][i]['categorie'] if results['metadatas'] else "Inconnue"
                })
        return formatted_results
    
if __name__ == "__main__":
    # 1. On charge la base existante (sans lui passer de chunks pour vérifier qu'elle recharge bien le disque)
    print("Chargement de la base vectorielle...")
    db = VectorDB(chunks=None)
    
    # 2. Liste des 5 questions de test
    questions_test = [
        "Quelle est la couleur du chat de Bob ?", # Présent dans le corpus, test direct
        "Comment s'appelle le félin bleu ?",  # Présent dans le corpus, test sémantique
        "Qu'est-ce qui est interdit de manger le mardi ?", # Présent dans le corpus, test direct
        "Comment fait-on pour entrer secrètement dans le laboratoire ?", # Présent dans le corpus, test sémantique
        "Quelle est la capitale du Japon ?"       # Test hors-sujet
    ]
    
    # 3. Exécution des tests
    print("\nTest de récupération des chunks les plus proches pour chaque question :")
    for i, question in enumerate(questions_test, 1):
        print(f"\nQuestion {i}: '{question}'")
        
        chunks_retrived = db.retrieve(question, n=3)
        
        # Affichage des résultats
        for rang, chunk in enumerate(chunks_retrived, 1):
            print(f"  -> Rang {rang} : [{chunk['source']}] {chunk['text']}")