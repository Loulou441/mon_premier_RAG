import os
from dotenv import load_dotenv
from database import VectorDB
from utils import csv_to_json
import json

# Chargement explicite de la clé API depuis le fichier .env [cite: 29, 74]
load_dotenv()

# Corpus imposé par le sujet
with open("data/05_corpus_rag.json", "r", encoding="utf-8") as f:
    CORPUS_TP = json.load(f)


def main():
    #1 : Initialisation de la base vectorielle
    # Au premier lancement, elle se crée. Au second, elle se rechargera sans réindexer !
    db = VectorDB(chunks = CORPUS_TP)

if __name__ == "__main__":
    main()