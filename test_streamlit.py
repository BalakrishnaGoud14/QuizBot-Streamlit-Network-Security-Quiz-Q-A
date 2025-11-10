#!/usr/bin/env python3
"""
Quick test to verify Streamlit app components
"""
import sys
from pathlib import Path

print(" Testing QuizBot Streamlit Components")
print("=" * 60)

# Test 1: Check imports
print("\n1️  Testing imports...")
try:
    import streamlit as st
    print("    Streamlit")
except ImportError as e:
    print(f"    Streamlit: {e}")
    sys.exit(1)

try:
    from langchain_community.llms.ollama import Ollama
    print("    LangChain Ollama")
except ImportError as e:
    print(f"    LangChain Ollama: {e}")

try:
    from langchain_community.vectorstores import Chroma
    print("    ChromaDB")
except ImportError as e:
    print(f"    ChromaDB: {e}")

# Test 2: Check Ollama connection
print("\n2️  Testing Ollama connection...")
try:
    import subprocess
    result = subprocess.run(['pgrep', '-fl', 'ollama'], capture_output=True, text=True)
    if result.stdout:
        print("    Ollama is running")
    else:
        print("     Ollama might not be running")
except Exception as e:
    print(f"     Could not check Ollama status: {e}")

# Test 3: Check models
print("\n3️  Testing Ollama models...")
try:
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
    if 'llama3.2' in result.stdout:
        print("    llama3.2 model found")
    else:
        print("    llama3.2 model not found")
    
    if 'nomic-embed-text' in result.stdout:
        print("    nomic-embed-text model found")
    else:
        print("    nomic-embed-text model not found")
except Exception as e:
    print(f"     Could not check models: {e}")

# Test 4: Check database
print("\n4️  Testing database...")
chroma_path = Path("nsrag/chroma")
if chroma_path.exists():
    print(f"    Database directory exists")
    sqlite_file = chroma_path / "chroma.sqlite3"
    if sqlite_file.exists():
        size_mb = sqlite_file.stat().st_size / (1024 * 1024)
        print(f"    SQLite database found ({size_mb:.2f} MB)")
    else:
        print("     SQLite database not found")
else:
    print("    Database directory not found")

# Test 5: Check data files
print("\n5️  Testing data files...")
data_path = Path("nsrag/data")
if data_path.exists():
    pdf_files = list(data_path.glob("*.pdf"))
    print(f"    Data directory exists ({len(pdf_files)} PDF files)")
else:
    print("     Data directory not found")

# Test 6: Try loading model
print("\n6️  Testing model loading...")
try:
    model = Ollama(model="llama3.2:latest")
    print("    Model initialized successfully")
    
    # Try a simple query
    response = model.invoke("Say 'test' in one word")
    print(f"    Model responding: {response[:50]}")
except Exception as e:
    print(f"    Model loading failed: {e}")

print("\n" + "=" * 60)
print(" Component testing complete!")
print("\n If all tests passed, run: streamlit run streamlit_app.py")
print(" If tests failed, check the error messages above")
