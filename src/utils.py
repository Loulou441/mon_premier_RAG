"""
Utility functions for the RAG system.
"""
import json
import csv

def csv_to_json(csv_file_path, json_file_path):
    # 1. Lire le fichier CSV
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        # DictReader transforme chaque ligne en dictionnaire { 'colonne': 'valeur' }
        csv_reader = csv.DictReader(csv_file)
        
        # On convertit l'itérateur en une vraie liste de dictionnaires
        data = list(csv_reader)

    # 2. Écrire le fichier JSON
    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        # indent=4 permet d'avoir un fichier JSON bien lisible et espacé
        # ensure_ascii=False permet de conserver les accents français
        json.dump(data, json_file, indent=4, ensure_ascii=False)