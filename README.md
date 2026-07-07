# Mon Premier RAG ⚡ (Pipeline Modulaire & Sécurisé avec Groq & ChromaDB)

Ce dépôt contient l'implémentation complète et reproductible du mini-TP guidé de construction d'un pipeline **RAG (Retrieval-Augmented Generation)**. Conçu sans frameworks d'orchestration lourds (sans LangChain ni LlamaIndex), ce projet met en évidence les mécanismes fondamentaux de l'ingestion, de la recherche sémantique (Retrieval), de l'atténuation des hallucinations et de la sécurité des prompts.

Le système est testé sur un **corpus de 200 chunks**, garantissant que les réponses correctes proviennent exclusivement du processus de récupération de documents et non de la mémoire pré-entraînée du LLM.

## ⚙️ Architecture du Projet

Le projet adopte une approche orientée objet avec une séparation stricte entre la logique d'exécution, la configuration et les instructions (prompts) :

```
mon_premier_RAG/
├── src/
│   ├── config.py                       # Configuration centralisée (modèles, .env, chemins)
│   ├── agent.py                        # Classe de base : client Groq unifié
│   ├── moderator_agent.py              # Agent de sécurité anti prompt-injection
│   ├── rag_orchestrator.py             # Chef d'orchestre du pipeline complet
│   ├── database.py                     # Gestion de la base vectorielle ChromaDB
│   ├── utils.py                        # Conversion CSV → JSON du corpus
│   └── prompts/
│       ├── moderator_prompt.txt        # Consignes de l'agent modérateur
│       └── rag_orchestrator_prompt.txt # Consignes de l'agent de génération
├── data/
│   ├── 05_corpus_rag.csv               # Corpus source (200 chunks)
│   └── 05_corpus_rag.json              # Corpus converti au format JSON
├── requirements.txt
├── LICENSE                             # Licence Apache 2.0
└── README.md
```

* **`config.py`** : Centralisation de la configuration (modèles, variables d'environnement, paramètres des dossiers).
* **`agent.py`** : Base commune instanciant le client officiel `Groq` de manière unifiée.
* **`database.py`** : Gestion du cycle de vie de la base vectorielle persistante **ChromaDB** et de l'encodage local avec normalisation spatiale via `sentence-transformers`.
* **`moderator_agent.py`** : Agent de sécurité interceptant les attaques par injection de prompt avant la phase de génération, forçant un format de sortie JSON strict.
* **`rag_orchestrator.py`** : Chef d'orchestre assemblant le flux complet (Modération → Retrieval → Augmentation → Génération).
* **`prompts/`** : Isolation complète des invites systèmes au format `.txt` (`rag_orchestrator_prompt.txt` et `moderator_prompt.txt`).
* **`utils.py`** : Outils de transformation de données (conversion automatique de données CSV vers JSON).

### Flux d'exécution (`RAGSystem.answer_question`)

```
Question utilisateur
        │
        ▼
1. Modération  ──── ModeratorAgent (Llama 3.3 70B, JSON strict)
        │                  │
        │           injection détectée ? ──► Oui ──► Réponse de refus
        │                  │
        ▼                 Non
2. Retrieval  ──── VectorDB.retrieve() (ChromaDB, top-3 chunks)
        │
        ▼
3. Augmentation ── Injection des chunks dans rag_orchestrator_prompt.txt
        │
        ▼
4. Génération ──── LLM Groq (Llama 3.3 70B) → réponse sourcée
```

---

## 🧠 Modèles utilisés

| Rôle | Modèle | Fournisseur |
|---|---|---|
| Embeddings (indexation & recherche) | `distiluse-base-multilingual-cased-v2` | `sentence-transformers` (local) |
| Génération de réponse | `llama-3.3-70b-versatile` | Groq API |
| Modération / anti prompt-injection | `llama-3.3-70b-versatile` | Groq API |

---

## 📚 Corpus de test

Le corpus (`data/05_corpus_rag.csv` / `.json`) est volontairement **absurde et fictif** (couleurs d'animaux improbables, règles inventées, biographies imaginaires...) afin de garantir qu'une bonne réponse ne peut provenir **que** du mécanisme de retrieval, et jamais des connaissances déjà apprises par le LLM.

* **200 chunks** au total, chacun structuré avec `id`, `text`, `source`, `categorie`.
* **26 sources** différentes (carnets personnels de personnages fictifs, registres, gazettes, annuaires, chroniques, etc.) : `carnet_de_bob`, `carnet_de_carla`, `carnet_de_diego`, `carnet_de_fatou`, `carnet_de_karim`, `carnet_de_leon`, `carnet_de_sophie`, `registre_animalier`, `annales_sportives`, `annuaire_metiers`, `archives_mairie`, `biographies_village`, `calendrier_festif`, `catalogue_objets`, `chroniques_artistiques`, `chroniques_village`, `commerce_local`, `gazette_locale`, `geographie_imaginaire`, `herbier_absurde`, `inventaire_bob`, `recensement_villebrume`, `recueil_culinaire`, `reglement_municipal`, `traditions_orales`, `traite_savant`.
* **16 catégories** thématiques, par exemple :

| Catégorie | Nb de chunks |
|---|---|
| divers | 31 |
| animaux | 22 |
| lieux | 15 |
| biographies | 12 |
| objets | 10 |
| cuisine | 10 |
| metiers | 10 |
| regles | 10 |
| traditions | 10 |
| plantes | 10 |
| sciences | 10 |
| art | 10 |
| sports | 10 |
| faits_divers | 10 |
| relations | 10 |
| chiffres | 10 |

---

## 🎯 Garanties de Reproductibilité

Pour assurer une exécution strictement identique sur n'importe quel poste de travail (macOS, Linux, Windows), les mécanismes suivants ont été blindés :

1.  **Isolation des Chemins (Robustesse OS)** : L'utilisation de `pathlib.Path` et de `os.path.join` garantit la résolution automatique des fichiers de prompts et de la base de données, peu importe le dossier depuis lequel le script est exécuté.
2.  **Gestion Dynamique des Embeddings** : Le modèle d'embedding est stocké dans les métadonnées de la collection ChromaDB à sa création. Si la base est rechargée depuis le disque, elle force l'utilisation du bon modèle, évitant les corruptions de dimensions vectorielles en cas de changement de configuration.
3.  **Contrôle Strict des Dépendances** : Verrouillage de la version de NumPy (`<2.0.0`) et de `sentence-transformers` (`==3.0.1`) pour éviter les incompatibilités ABI classiques lors des nouvelles installations de ChromaDB.

---

## 🛡️ Sécurité : détection des injections de prompt

Avant toute étape de retrieval ou de génération, chaque question passe par le `ModeratorAgent`, qui :

* renvoie systématiquement un objet JSON strict `{"prompt_injection": bool, "raison": str}` (`response_format="json_object"`, `temperature=0`) ;
* est calibré pour ne bloquer que les tentatives ciblant explicitement le comportement de l'agent (ex. « ignore tes instructions », « oublie le format JSON », demande de révéler le prompt système, injection de fausses balises système, demande d'exécuter une action externe) ;
* laisse volontairement passer le langage parlé naturel d'une transcription (impératifs, apartés, consignes adressées à des tiers) qui ne visent pas l'agent lui-même ;
* résout le doute en faveur du contenu légitime (`prompt_injection = false`) en cas d'ambiguïté réelle.

Si une injection est détectée, le pipeline s'arrête immédiatement et renvoie un message de refus, sans jamais atteindre l'étape de génération.

---

## 🚀 Installation & Déploiement Rapide

Suivez scrupuleusement ces étapes pour reproduire l'environnement d'exécution.

### 1. Clonage et Navigation
```bash
git clone https://github.com/VOTRE-COMPTE/mon_premier_RAG.git
cd mon_premier_RAG
```

### 2. Création de l'environnement virtuel
```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```

### 3. Installation des dépendances
```bash
pip install -r requirements.txt
```

### 4. Configuration de la clé API Groq
Créez un fichier `.env` à la racine du projet et ajoutez-y votre clé (obtenue sur [console.groq.com](https://console.groq.com)) :
```
GROQ_API_KEY=votre_cle_ici
```
> ⚠️ Sans cette variable, `config.py` lève volontairement une `ValueError` explicite au démarrage.

### 5. (Optionnel) Régénération du corpus JSON depuis le CSV
```bash
python src/utils.py
```

### 6. Initialisation de la base vectorielle
La base ChromaDB (`src/chroma_db/`) est créée automatiquement au premier lancement si elle n'existe pas encore, à partir des chunks chargés depuis `data/05_corpus_rag.json`.

---

## ▶️ Utilisation

### Interroger le pipeline RAG complet
```bash
cd src
python rag_orchestrator.py
```
Ce script initialise (ou recharge) la base vectorielle, instancie `RAGSystem`, puis exécute plusieurs questions d'exemple illustrant :
* une question dans le périmètre du corpus,
* une question hors périmètre,
* une tentative de contournement (données sensibles) bloquée par le modérateur.

### Utilisation programmatique
```python
from database import VectorDB
from rag_orchestrator import RAGSystem

db = VectorDB()  # charge la base existante ou la crée depuis le corpus
rag = RAGSystem(db)

reponse = rag.answer_question("Quelle est la couleur du chat de Bob ?")
print(reponse)
```

### Tester uniquement le retrieval
```bash
python database.py
```

### Tester uniquement la modération
```bash
python moderator_agent.py
```

---

## 📦 Dépendances principales

```
groq
sentence-transformers==3.0.1
chromadb
python-dotenv
numpy<2.0.0
```

---

## 📄 Licence

Ce projet est distribué sous licence **Apache License 2.0**. Voir le fichier [`LICENSE`](./LICENSE) pour le texte complet.