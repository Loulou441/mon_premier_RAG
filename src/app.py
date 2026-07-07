"""
Interface Streamlit pour le projet "Mon Premier RAG".

Lancement :
    cd src
    streamlit run streamlit_app.py
"""
import streamlit as st

from config import COLLECTION_NAME, EMBEDDING_MODEL_NAME, LLM_GENERATION_MODEL
from database import VectorDB
from rag_orchestrator import RAGSystem


# --------------------------------------------------------------------------- #
# Configuration de la page
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Mon Premier RAG",
    page_icon="⚡",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Chargement des ressources (mise en cache pour ne pas tout recharger
# à chaque interaction utilisateur)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Chargement de la base vectorielle...")
def load_rag_system() -> RAGSystem:
    db = VectorDB()
    return RAGSystem(db)


try:
    rag_system = load_rag_system()
    load_error = None
except Exception as exc:  # noqa: BLE001
    rag_system = None
    load_error = str(exc)


# --------------------------------------------------------------------------- #
# État de session
# --------------------------------------------------------------------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []  # liste de dicts {role, content, chunks?, moderation?}


# --------------------------------------------------------------------------- #
# Barre latérale
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("⚡ Mon Premier RAG")
    st.caption("Pipeline RAG modulaire & sécurisé — Groq & ChromaDB")

    st.divider()
    st.subheader("Configuration")
    st.markdown(f"**Collection** : `{COLLECTION_NAME}`")
    st.markdown(f"**Embeddings** : `{EMBEDDING_MODEL_NAME}`")
    st.markdown(f"**Génération** : `{LLM_GENERATION_MODEL}`")

    if rag_system is not None:
        try:
            nb_chunks = rag_system.db.collection.count()
            st.markdown(f"**Chunks indexés** : `{nb_chunks}`")
        except Exception:
            pass

    st.divider()
    debug_mode = st.toggle(
        "Mode debug",
        value=False,
        help="Affiche les chunks récupérés et le résultat brut de la modération pour chaque question.",
    )

    n_chunks = st.slider(
        "Nombre de chunks à récupérer (n)",
        min_value=1,
        max_value=10,
        value=3,
        help="Nombre de passages les plus pertinents transmis au LLM pour générer la réponse.",
    )

    st.divider()
    if st.button("🗑️ Réinitialiser la conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    with st.expander("À propos du corpus"):
        st.markdown(
            "Le corpus contient **200 chunks fictifs** répartis en "
            "16 catégories (animaux, lieux, biographies, sciences, "
            "cuisine, sports...) et 26 sources. Il est volontairement "
            "absurde pour garantir que les réponses proviennent "
            "uniquement du retrieval, jamais des connaissances "
            "pré-entraînées du LLM."
        )


# --------------------------------------------------------------------------- #
# Corps principal
# --------------------------------------------------------------------------- #
st.title("💬 Interroger le corpus")

if load_error:
    st.error(
        "Impossible d'initialiser le système RAG. Vérifiez que la clé "
        f"`GROQ_API_KEY` est bien présente dans votre fichier `.env`.\n\n"
        f"Détail : {load_error}"
    )
    st.stop()

st.caption(
    "Posez une question sur le corpus. Chaque requête passe d'abord par un "
    "agent de modération anti prompt-injection avant d'être traitée."
)

# Affichage de l'historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and debug_mode:
            if msg.get("moderation") is not None:
                with st.expander("🛡️ Résultat de la modération"):
                    st.json(msg["moderation"])
            if msg.get("chunks"):
                with st.expander(f"📄 Chunks récupérés ({len(msg['chunks'])})"):
                    for i, chunk in enumerate(msg["chunks"], 1):
                        st.markdown(
                            f"**{i}. [{chunk['source']}]** "
                            f"*(catégorie : {chunk['categorie']})*\n\n{chunk['text']}"
                        )
                        st.divider()

# Zone de saisie
question = st.chat_input("Votre question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Modération de la requête..."):
            moderation_result = rag_system.moderator.moderate_transcript(question)

        if moderation_result.get('prompt_injection'):
            answer = (
                "Désolé, votre requête a été rejetée par notre agent de "
                f"sécurité (Tentative d'injection détectée : "
                f"{moderation_result.get('raison', 'raison non précisée')})."
            )
            matched_chunks = []
            st.warning(answer)
        else:
            with st.spinner("Recherche des passages pertinents..."):
                matched_chunks = rag_system.db.retrieve(question, n=n_chunks)

            chunks_text = ""
            for idx, chunk in enumerate(matched_chunks):
                chunks_text += (
                    f"\n[Document {idx + 1}] (Source: {chunk['source']})\n"
                    f"{chunk['text']}\n (Categorie: {chunk['categorie']})"
                )
            completed_system_prompt = rag_system.rag_template.replace(
                "{{Chunks}}", chunks_text
            )

            with st.spinner("Génération de la réponse..."):
                response = rag_system.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": completed_system_prompt},
                        {"role": "user", "content": question},
                    ],
                    model=LLM_GENERATION_MODEL,
                )
                answer = response.choices[0].message.content

            st.markdown(answer)

        if debug_mode:
            with st.expander("🛡️ Résultat de la modération"):
                st.json(moderation_result)
            if matched_chunks:
                with st.expander(f"📄 Chunks récupérés ({len(matched_chunks)})"):
                    for i, chunk in enumerate(matched_chunks, 1):
                        st.markdown(
                            f"**{i}. [{chunk['source']}]** "
                            f"*(catégorie : {chunk['categorie']})*\n\n{chunk['text']}"
                        )
                        st.divider()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "chunks": matched_chunks,
            "moderation": moderation_result,
        }
    )