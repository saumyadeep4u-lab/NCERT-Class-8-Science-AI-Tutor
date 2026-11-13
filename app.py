import os
import csv
from datetime import datetime
import re

import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain_core.prompts import ChatPromptTemplate

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="NCERT Class 8 Science AI Tutor",
    page_icon="📘",
    layout="wide"
)

st.markdown("""
# 📘 Curriculum-Based AI Tutor

### NCERT Class 8 Science Question Answering System

Ask questions from the NCERT Class 8 Science textbook.
""")

# --------------------------------------------------
# LOGGING FUNCTION
# --------------------------------------------------

def log_interaction(
    question,
    answer,
    similarity_score,
    retrieved_docs
):

    log_file = "user_interaction_logs.csv"

    file_exists = os.path.isfile(log_file)

    with open(
        log_file,
        mode="a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if not file_exists:

            writer.writerow([
                "timestamp",
                "question",
                "answer",
                "similarity_score",
                "retrieved_docs"
            ])

        writer.writerow([
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            question,
            answer,
            similarity_score,
            retrieved_docs
        ])

# --------------------------------------------------
# LOAD VECTORSTORE
# --------------------------------------------------

@st.cache_resource
def load_vectorstore():

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    vectorstore_path = os.path.join(
        base_dir,
        "notebooks",
        "class8_vectorstore"
    )

    if not os.path.exists(vectorstore_path):

        st.error(
            f"Vectorstore folder not found:\n{vectorstore_path}"
        )

        st.stop()

    return FAISS.load_local(
        vectorstore_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )

# --------------------------------------------------
# LOAD LLM
# --------------------------------------------------

@st.cache_resource
def load_llm():

    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    model_path = os.path.join(
        base_dir,
        "models",
        "llama-2-7b-chat.Q4_K_M.gguf"
    )

    if not os.path.exists(model_path):

        st.error(
            f"Model file not found:\n{model_path}"
        )

        st.stop()

    llm = CTransformers(
        model=model_path,
        model_type="llama",
        config={
            "max_new_tokens": 120,
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": 1.15,
            "context_length": 4096
        }
    )

    return llm

# --------------------------------------------------
# LOAD COMPONENTS
# --------------------------------------------------

with st.spinner("Loading vector database..."):
    vectorstore = load_vectorstore()

with st.spinner("Loading language model..."):
    llm = load_llm()

# --------------------------------------------------
# PROMPT TEMPLATE
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template("""
You are an NCERT Class 8 Science tutor.

Rules:
- Answer ONLY using the provided context.
- Do not use outside knowledge.
- Keep answers short and textbook-like.
- Use simple language suitable for Class 8 students.
- Maximum 3 sentences.
- If the answer is not available in the context, say:
  "I'm focused on Class 8 Science NCERT content."
- Do not hallucinate.

Context:
{context}

Question:
{input}

Answer:
""")

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------------------------------------------------
# QUESTION INPUT
# --------------------------------------------------

query = st.text_input(
    "Ask your question:",
    placeholder="Example: What is photosynthesis?"
)

# --------------------------------------------------
# SUBMIT BUTTON
# --------------------------------------------------

if st.button("Submit"):

    if not query.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Generating answer..."
        ):

            results = vectorstore.similarity_search_with_score(
                query,
                k=3
            )

            similarity_score = None
            retrieved_docs = 0

            if len(results) == 0:

                answer = (
                    "I'm focused on Class 8 Science NCERT content."
                )

                sources = []

            else:

                best_score = results[0][1]

                similarity_score = round(
                    float(best_score),
                    4
                )

                # ----------------------------------
                # Keyword overlap check
                # ----------------------------------

                stop_words = {
                    "what", "who", "when", "where", "why",
                    "how", "is", "are", "was", "were",
                    "the", "a", "an", "of", "in", "on",
                    "for", "to", "and", "or"
                }

                query_words = {
                    re.sub(r'[^a-zA-Z0-9]', '', word.lower())
                    for word in query.split()
                }

                query_words = {
                    word
                    for word in query_words
                    if len(word) > 2 and word not in stop_words
                }

                keyword_found = False

                for doc, score in results:

                    content_words = set(
                        re.findall(
                            r'\b\w+\b',
                            doc.page_content.lower()
                        )
                    )

                    if query_words.intersection(content_words):
                        keyword_found = True
                        break

# ----------------------------------
# Out-of-syllabus detection
# ----------------------------------

                if best_score > 1.3 or not keyword_found:

                    answer = (
                        "I'm focused on Class 8 Science NCERT content, try re-phrasing."
                )

                    sources = []

                else:

                    docs = [
                        doc
                        for doc, score in results
                    ]

                    retrieved_docs = len(docs)

                    context = "\n\n".join(
                        [
                            doc.page_content
                            for doc in docs
                        ]
                    )

                    formatted_prompt = prompt.format(
                         context=context,
                         input=query
                    )

                    answer = llm.invoke(
                         formatted_prompt
                    )

                    sources = docs
# --------------------------------------
# STORE CHAT HISTORY
# --------------------------------------

            st.session_state.chat_history.append({

                "question": query,
                "answer": answer,
                "sources": sources

            })

            # --------------------------------------
            # LOG INTERACTION
            # --------------------------------------

            log_interaction(
                question=query,
                answer=answer,
                similarity_score=similarity_score,
                retrieved_docs=retrieved_docs
            )

# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

if st.session_state.chat_history:

    st.markdown("---")

    st.markdown("## Conversation")

    for item in reversed(
        st.session_state.chat_history
    ):

        st.markdown(
            f"### ❓ Question\n{item['question']}"
        )

        st.markdown(
            f"### ✅ Answer\n{item['answer']}"
        )

        if item["sources"]:

            with st.expander(
                "📚 View Source Snippets"
            ):

                for i, doc in enumerate(
                    item["sources"]
                ):

                    st.markdown(
                        f"**Source {i+1}**"
                    )

                    st.write(
                        doc.page_content[:500]
                    )

                    st.divider()