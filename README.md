# 📚 Curriculum-Based AI Tutor for NCERT Class 8 Science (RAG)

An end-to-end **Retrieval-Augmented Generation (RAG)** AI Tutor engineered to answer student questions strictly grounded in the **NCERT Class 8 Science** textbook. Built with **LangChain**, **FAISS**, **Sentence Transformers (`all-MiniLM-L6-v2`)**, and **Llama-2-7B-Q Chat (GGUF)**, and deployed via an interactive **Streamlit** web interface with transparent textbook source snippets and out-of-syllabus guardrails.

---

## 🚀 Key Features
- 🔍 **Semantic Textbook Retrieval:** Converts textbook passages and student queries into vector embeddings using `all-MiniLM-L6-v2` and retrieves top-matching context via a **FAISS** vector database.
- 🤖 **Grounded Answer Generation:** Supplies retrieved NCERT passages to **Llama-2-7B-Q Chat (GGUF)** so responses remain faithful to the Class 8 syllabus.
- 📖 **Source Snippet Transparency:** Displays the exact NCERT textbook snippets used to formulate each answer.
- 🛡️ **Out-of-Syllabus Guardrails:** Combines retrieval similarity scores and query-word matching to block off-topic queries (e.g., *"What is blockchain?"*, *"What is ChatGPT?"*), returning:
  > *"I'm focused on Class 8 Science NCERT content, try re-phrasing."*
- 📊 **Comprehensive Evaluation:** Benchmarked across 20 syllabus and out-of-syllabus test queries using **BLEU**, **ROUGE-L**, and human evaluation criteria (Correctness, Completeness, Relevance, and NCERT Alignment).
- 📝 **Automated Interaction Logging:** Records student queries, retrieved contexts, and tutor responses in `user_interaction_logs.csv` and `self_interaction_logs.csv`.

---

## 🔄 System Architecture & RAG Workflow

`User Question` ➔ `Embedding Generation (all-MiniLM-L6-v2)` ➔ `FAISS Vector Retrieval` ➔ `Out-of-Syllabus Check` ➔ `Context Construction + Llama-2-7B-Q` ➔ `Generated Answer + NCERT Source Snippets`

1. **Dataset Preparation (`data/`):** Extracted and cleaned the NCERT Class 8 Science textbook (`class8_science.txt`), splitting text into overlapping chunks (`chunk_size=500`, `chunk_overlap=50`) via `RecursiveCharacterTextSplitter` and saving to `class8_science.jsonl`.
2. **Vector Indexing (`models/` & `notebooks/class8_vectorstore/`):** Encoded text chunks into dense embeddings using `all-MiniLM-L6-v2` and indexed them in FAISS (`index.faiss`, `index.pkl`, `science_index.faiss`).
3. **Retrieval & Generation (`notebooks/ai_tutor_class8.ipynb` & `app.py`):** Matched user queries against stored vectors, filtered out-of-syllabus queries, and prompted `Llama-2-7B-Q` to generate student-friendly explanations.

---

## 🛠️ Technologies Used
- **Language Model:** Llama-2-7B-Q Chat (GGUF) (`llama-2-7b-chat.Q4_K_M`)
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Database:** FAISS (`faiss-cpu`)
- **Frameworks & Libraries:** LangChain, Streamlit, Sentence Transformers, Pandas, NumPy
- **Evaluation Metrics:** BLEU, ROUGE-L, Human Rubric Review (`evaluation.csv`)

---

## 📈 Evaluation & Key Findings
- **Syllabus Mastery:** Accurately answered curriculum questions covering cells, microorganisms, force, plant cells, the nucleus, and scientific investigation.
- **100% Out-of-Syllabus Rejection:** Successfully filtered out non-curriculum topics including Machine Learning, ChatGPT, Blockchain, Quantum Computing, and AI in rockets.
- **Metric Insights (BLEU vs. ROUGE-L & Human Review):** While short reference answers caused lower exact-match BLEU scores on detailed explanations, strong **ROUGE-L** scores and human reviews confirmed high semantic accuracy, completeness, and pedagogical value.

---

## 📂 Project Structure

```text
Curriculum-Based-AI-Tutor-for-NCERT-Class-8-Science/
├── data/
│   ├── processed/
│   │   └── class8_science.jsonl                         # Chunked NCERT textbook data (size=500, overlap=50)
│   └── raw/
│       └── class8_science.txt                           # Raw extracted NCERT Class 8 Science text
├── models/
│   ├── llama-2-7b-chat.Q4_K_M.txt                       # Model reference / configuration info for Llama-2 GGUF
│   └── science_index.faiss                              # Compiled FAISS vector index
├── notebooks/
│   ├── class8_vectorstore/
│   │   ├── index.faiss                                  # Serialized FAISS vector store
│   │   └── index.pkl                                    # Document metadata mapping for FAISS
│   ├── ai_tutor_class8.ipynb                            # End-to-end RAG indexing, inference & evaluation notebook
│   ├── evaluation.csv                                   # BLEU, ROUGE-L & human evaluation benchmark results
│   └── self_interaction_logs.csv                        # Notebook test interaction logs
├── Brief PDF Report on a Curriculum-Based AI T...pdf    # Detailed technical project report
├── app.py                                               # Interactive Streamlit AI Tutor web application
├── user_interaction_logs.csv                            # Logged student Q&A sessions from the Streamlit app
├── requirements.txt                                     # Python dependencies
└── README.md                                            # Project documentation
```

---

## 🔮 Future Scope
- Enhanced retrieval precision using Cross-Encoder reranking models
- Multilingual support for regional Indian languages
- Integration of larger / cloud-hosted LLMs and voice-based interaction
- Automated scientific diagram generation and personalized learning paths
- Multi-grade NCERT curriculum expansion (Classes 6 to 12)

---

## 💻 How to Run Locally

1. **Clone the repository:** `git clone [(https://github.com/saumyadeep4u-lab/NCERT-Class-8-Science-AI-Tutor.git)]`
2. **Open the project folder:** `cd Curriculum-Based-AI-Tutor-for-NCERT-Class-8-Science`
3. **Install dependencies:** `pip install -r requirements.txt`
4. **Download Model Weights:** Place the quantized `llama-2-7b-chat.Q4_K_M.gguf` model file inside the `models/` directory.
5. **Launch the Streamlit app:** `streamlit run app.py`

---

## 👨‍💻 Developer
**Saumyadeep Bardhan**
