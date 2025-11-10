# QuizBot — LLM-Powered Network Security Quiz Generator

## 📘 Project Description

QuizBot is an AI-powered educational tool designed to help students master **Network Security concepts**.  
It automatically generates quiz questions (MCQ, True/False, Short Answer) from uploaded course materials and contextual embeddings.

This project uses **local LLMs**, **ChromaDB**, and **embedding-based RAG** to produce context-aware security quizzes, enhancing learning and revision efficiency.

---

## 📚 Reference Documentation

- **Ollama local LLM runtime:** https://ollama.ai/

---

## 🏗️ System Architecture

### 1️⃣ Data Processing Layer

- **PDF/Text Loader** — Import lecture slides or notes
- **Text Preprocessing** — Chunk text into learning units
- **Embedding Model** — Generate embeddings using `nomic-embed-text` (Ollama)

### 2️⃣ Vector Storage Layer

- **Embedding Store:** `ChromaDB` local vector DB (`./chroma/`)
- **Persistent Cache:** JSON + Chroma storage

### 3️⃣ Retrieval + LLM Layer

- **RAG Pipeline**
- **Local LLM (Llama models)**
- **Context-aware question generation**

### 4️⃣ Interface Layer

- **Streamlit UI** (`streamlit_app.py` / `streamlit_simple.py`)
- Upload materials, generate quiz, receive feedback

---

## 📁 Directory Structure

```
QuizBot/
 ├── streamlit_app.py
 ├── streamlit_simple.py
 ├── chroma/
 ├── fix_database.py
 ├── test_streamlit.py
 ├── requirements.txt
 └── README.md
```

---

## ✅ Prerequisites

| Component | Requirement                  |
| --------- | ---------------------------- |
| Python    | 3.9+                         |
| Ollama    | Installed + running          |
| Models    | `nomic-embed-text`, `llama3` |
| OS        | Linux / macOS / Windows WSL  |

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Project

```bash
git clone <repo-link>
cd QuizBot-main/QuizBot-main
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Create Embedding Database

```bash
python fix_database.py
```

---

## 🚀 Running QuizBot

### Start Ollama Service

```bash
ollama serve
```

### Launch Streamlit UI

```bash
streamlit run streamlit_simple.py
```

or

```bash
streamlit run streamlit_app.py
```

## 🧠 Example Output

```
Question: What does a MAC ensure?
1. Integrity
2. Confidentiality
...
Answer: 1



---

## 🔧 Troubleshooting
| Issue | Fix |
|---|---|
Ollama not running | `ollama serve` |
DB issues | delete `chroma/` & run `fix_database.py` |
Streamlit error | `pip install streamlit` |
Embedding issues | ensure `nomic-embed-text` installed |

---



## 🎓 Summary
✔ Local & private RAG quiz generator
✔ Network-security focused
✔ Streamlit interactive UI

---

```
