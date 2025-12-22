
***

# QuizBot – Streamlit Network Security Quiz & Q&A

This project is a simplified version of **QuizBot**, an AI-powered tool for generating network security quizzes and answering questions using local language models. It provides a Streamlit web UI for quiz generation and Q&A, backed by either a direct PDF reader or a full retrieval-augmented generation (RAG) pipeline.[1][2][3]

## Features

- **Interactive quizzes:** Multiple-choice and True/False questions on core network security topics (e.g., RSA, TLS, hash functions, Diffie–Hellman).[2][1]
- **Q&A chat:** Ask arbitrary questions about network security and get detailed answers from a local LLM with optional document context.[3][1][2]
- **Beautiful Streamlit UI:** Gradient header, card-style questions, colored feedback for correct and wrong answers.[1][2]
- **Local models:** Uses Ollama-hosted models (e.g., `llama3.2:latest`) so everything runs on your machine.[2][3][1]

## Components

- `streamlit_simple.py` – Simplified Streamlit app for quiz generation and Q&A.[1][2]
- `nsrag/rag.py` – RAG module for PDF indexing, semantic search, web fallback (DuckDuckGo), and LLM answer generation.[3]
- `nsrag/get_embedding_function.py` – Helper for creating embedding functions using Ollama embeddings.[4]
- `requirements.txt` – Python dependencies, including Streamlit, sentence-transformers, langchain-community, PyPDF, BeautifulSoup, and more.[5]

Depending on how you wire things, `streamlit_simple.py` can either:[2][1]

- Use direct PDF loading with `PyPDFDirectoryLoader`, or  
- Call into `nsrag.rag` (for `answerquestion`, `buildvectorstore`) for full RAG behavior.  

## Prerequisites

- Python 3.8+.[5]
- `pip` for installing dependencies.[5]
- **Ollama** installed and running locally, with at least `llama3.2:latest` downloaded.[3][1][2]
- (Optional but recommended) A directory of PDF lecture notes under `nsrag/data/` for richer context.[1][3]

### Install dependencies

From the project root:

```bash
pip install -r requirements.txt
```


### Start Ollama

- Launch the Ollama service/app.  
- Optionally set environment variables for custom models:

```bash
export OLLAMAMODEL=llama3.2:latest
export EMBEDMODEL=all-MiniLM-L6-v2
```


## Running the Streamlit App

From the project root:

```bash
streamlit run streamlit_simple.py
```



### Main modes

In the sidebar you can choose:[1][2]

- **Generate Quiz:**  
  - Quiz type: MCQ or True/False.  
  - Topic mode: Random topics or a specific topic from the built-in `TOPICS` list (e.g., “RSA”, “TLS 1.0 Lucky 13 Attack”).    
- **Ask Questions:**  
  - Free-form questions like “Explain Diffie–Hellman key exchange” or “What is a length extension attack?”.  
- **About:**  
  - Shows basic stats: number of topics, model name, and interface count.  

When you click **Generate New Quiz**, the app:[2][1]

1. Builds a prompt using the configured topics and question type.  
2. Uses `Ollama` to generate raw quiz text with a network-security context.  
3. Parses the text into questions and options (MCQ or True/False).  

When you click **Submit Quiz**, the app:[1][2]

1. Sends your answers plus the original questions to the model with an evaluation prompt.  
2. Extracts the correct answers and explanations from the model output.  
3. Displays color-coded feedback and an overall score.  

The **Ask Questions** mode builds a focused prompt using a slice of the PDF context and asks the model to answer clearly.[2][1]

## RAG Backend (nsrag/rag.py)

The `rag.py` module provides a more advanced backend that:[3]

- Reads PDFs from a `data/` directory and chunks them into line-based segments.  
- Uses `sentence-transformers` (e.g., `all-MiniLM-L6-v2`) to embed chunks and persist them under `nsrag/vstore/`.  
- Runs semantic search over the embeddings to fetch the most relevant chunks for a query.  
- Optionally calls DuckDuckGo HTML search for web snippets when PDF evidence is weak.  
- Builds a prompt with PDF citations and web references and sends it to an Ollama model.  
- Returns the answer plus a list of sources (PDF page/line ranges and/or URLs).  

You can rebuild the vector store from the command line:[3]

```bash
cd nsrag
python3 rag.py --rebuild
```

And test a single question via CLI:  

```bash
python3 rag.py --question "Explain RSA key generation"
```


## Embeddings Helper

`get_embedding_function.py` provides a helper that returns an Ollama-based embedding function:[4]

- Uses `OllamaEmbeddings` with the `nomic-embed-text` model.  
- Designed to plug into LangChain components that accept a `embeddings` object.  

## Configuration and Customization

- **Topics list:** Edit `TOPICS` in `streamlit_simple.py` to add or remove network security topics.[1][2]
- **Models:**  
  - Change the default LLM via `OLLAMAMODEL` env var or by editing `getollamamodel` in `rag.py`.[3]
  - Change the sentence-transformer via `EMBEDMODEL` env var.[3]
- **PDF location:** By default, `rag.py` expects a `data/` directory next to `rag.py`; adjust `DATADIR` if needed.[3]


