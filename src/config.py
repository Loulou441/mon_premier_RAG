import os
from dotenv import load_dotenv

load_dotenv()  # Charger les variables d'environnement depuis le fichier .env
# Chemins de stockage
DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

# Modèles
EMBEDDING_MODEL_NAME = "distiluse-base-multilingual-cased-v2" # Donné dans le sujet
LLM_GENERATION_MODEL = "llama-3.3-70b-versatile"              # Donné dans le sujet
LLM_MODERATION_MODEL = "llama-3.3-70b-versatile"

#Clé GROQ
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

## Sécurité pour la clé GROQ
if GROQ_API_KEY is None:
    raise ValueError(
        "GROQ_API_KEY introuvable. Vérifier son existance dans .env"
        "et qu'il contient une ligne : GROQ_API_KEY=..."
    )

# Nom de collection
COLLECTION_NAME = "tp_rag_collection"
