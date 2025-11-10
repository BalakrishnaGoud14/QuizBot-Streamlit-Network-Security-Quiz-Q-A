
"""
Test script to verify QuizBot setup with llama3.2
"""
import warnings
warnings.filterwarnings("ignore")

print("Testing QuizBot Setup...")
print("-" * 50)

# Test 1: Check Ollama model
print("\n1. Testing Ollama llama3.2 model...")
try:
    from langchain_community.llms.ollama import Ollama
    model = Ollama(model="llama3.2:latest")
    response = model.invoke("What is encryption? Answer in one sentence.")
    print(f" Model loaded and responding")
    print(f"  Response: {response[:100]}...")
except Exception as e:
    print(f" Error: {e}")

# Test 2: Check embedding function
print("\n2. Testing embedding function...")
try:
    from get_embedding_function import get_embedding_function
    embedding_function = get_embedding_function()
    test_embedding = embedding_function.embed_query("test")
    print(f" Embeddings working (dimension: {len(test_embedding)})")
except Exception as e:
    print(f" Error: {e}")

# Test 3: Check ChromaDB
print("\n3. Testing ChromaDB connection...")
try:
    from langchain_community.vectorstores import Chroma
    CHROMA_PATH = "chroma"
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    count = db._collection.count()
    print(f" ChromaDB connected ({count} documents)")
except Exception as e:
    print(f" Error: {e}")

# Test 4: Check data directory
print("\n4. Checking data directory...")
import os
DATA_PATH = "data"
if os.path.exists(DATA_PATH):
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    print(f" Data directory exists ({len(pdf_files)} PDF files)")
else:
    print(f" Data directory not found")

print("\n" + "-" * 50)
print("Setup verification complete!")
print("\nYou can now run:")
print("  - python3 rag.py (for CLI interface)")
print("  - python3 testApi.py (for Flask API)")
