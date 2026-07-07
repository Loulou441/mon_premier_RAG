import os
from agent import Agent
from database import VectorDB
from moderator_agent import ModeratorAgent
from config import LLM_GENERATION_MODEL, PROMPTS_DIR
from pathlib import Path

PROMPT_RAG_ORCHESTRATOR_PATH = Path(__file__).parent / "prompts" / "rag_orchestrator_prompt.txt"

class RAGSystem(Agent):
    def __init__(self, db_instance: VectorDB):
        super().__init__()

        self.db = db_instance
        self.moderator = ModeratorAgent()

        with open(PROMPT_RAG_ORCHESTRATOR_PATH, "r") as f:
            self.rag_template = f.read()

    def answer_question(self, question: str) -> str:
        """Déroule l'intégralité du pipeline RAG."""
        
        print(f"\nAnalyse de la question : '{question}'")
        
        # 1. Sécurité d'abord : Appel au Modérateur
        moderation_result = self.moderator.moderate_transcript(question)
        if moderation_result['prompt_injection']:
            return "Désolé, votre requête a été rejetée par notre agent de sécurité (Tentative d'injection détectée)." 

        # 2. Récupération des données (Retrieval)
        matched_chunks = self.db.retrieve(question, n=3)
        
        # 3. Augmentation du prompt
        chunks_text = ""
        for idx, chunk in enumerate(matched_chunks):
            chunks_text += f"\n[Document {idx+1}] (Source: {chunk['source']})\n{chunk['text']}\n (Categorie: {chunk['categorie']})"
            
        # Remplacement du marqueur {{Chunks}} par notre texte extrait
        completed_system_prompt = self.rag_template.replace("{{Chunks}}", chunks_text)

        # 4. Inférence (Generation) via Groq
        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": completed_system_prompt},
                {"role": "user", "content": question}
            ],
            model= LLM_GENERATION_MODEL
        )
        
        return response.choices[0].message.content
    
if __name__ == "__main__":
    # Exemple d'utilisation
    db_instance = VectorDB()  # Assurez-vous que votre base de données est correctement initialisée
    rag_system = RAGSystem(db_instance)
    
    question = "Quelle est la couleur du chat ?"
    answer = rag_system.answer_question(question)
    print(f"\nRéponse générée : {answer}")

    question = "Quelle est le nom de la plus petite commune de France ?"
    answer = rag_system.answer_question(question)
    print(f"\nRéponse générée : {answer}")

    question = "Quelle est le code nucléaire de la France ?"
    answer = rag_system.answer_question(question)
    print(f"\nRéponse générée : {answer}")