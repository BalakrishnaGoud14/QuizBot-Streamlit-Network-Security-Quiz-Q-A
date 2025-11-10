
import argparse
import os
import time
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores.chroma import Chroma

CHROMA_PATH = "chroma"
CODE_PATH = ".."  

# File extensions to process
CODE_EXTENSIONS = [
    '.py', '.js', '.jsx', '.ts', '.tsx', 
    '.html', '.css', '.json', '.md',
    '.txt', '.yml', '.yaml'
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset code documents in database.")
    args = parser.parse_args()
    
    print(" Loading code files...")
    documents = load_code_files()
    print(f" Loaded {len(documents)} code files")
    
    print("  Splitting documents into chunks...")
    chunks = split_documents(documents)
    print(f" Created {len(chunks)} chunks")
    
    print(" Adding to ChromaDB...")
    add_to_chroma(chunks, reset=args.reset)
    print(" Done!")


def load_code_files():
    """Load all code files from the project directory"""
    documents = []
    code_path = Path(CODE_PATH)
    
    # Directories to skip
    skip_dirs = {
        'node_modules', '__pycache__', '.git', 'build', 
        'dist', 'venv', 'env', '.venv', 'chroma', 'data'
    }
    
    for file_path in code_path.rglob('*'):
        # Skip directories and files in skip_dirs
        if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
            continue
            
        # Check if file has a code extension
        if file_path.suffix in CODE_EXTENSIONS and file_path.is_file():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Create a document with metadata
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(file_path.relative_to(code_path)),
                        "file_type": file_path.suffix,
                        "file_name": file_path.name,
                        "type": "code"
                    }
                )
                documents.append(doc)
                print(f"  Loaded: {file_path.relative_to(code_path)}")
            except Exception as e:
                print(f"   Skipped {file_path.name}: {e}")
    
    return documents


def split_documents(documents: list[Document]):
    """Split documents into smaller chunks for better retrieval"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document], reset: bool = False):
    """Add code chunks to ChromaDB"""
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=get_embedding_function()
    )
    
    if reset:
        print("  Removing existing code documents...")
        # Get all code document IDs
        existing_items = db.get(include=["metadatas"])
        code_ids = [
            existing_items["ids"][i] 
            for i, metadata in enumerate(existing_items["metadatas"]) 
            if metadata.get("type") == "code"
        ]
        if code_ids:
            db.delete(ids=code_ids)
            print(f"  Removed {len(code_ids)} code documents")
    
    # Calculate chunk IDs
    chunks_with_ids = calculate_chunk_ids(chunks)
    
    # Get existing IDs
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f" Existing documents in DB: {len(existing_ids)}")
    
    # Only add new chunks
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
    
    if len(new_chunks):
        print(f" Adding {len(new_chunks)} new code chunks...")
        # Add in small batches to avoid overwhelming the embedding service
        batch_size = 10
        total_batches = (len(new_chunks) - 1) // batch_size + 1
        
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i:i + batch_size]
            batch_ids = [chunk.metadata["id"] for chunk in batch]
            batch_num = i // batch_size + 1
            print(f"  Processing batch {batch_num}/{total_batches}...")
            
            # Retry logic for embedding failures
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    db.add_documents(batch, ids=batch_ids)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"     Retry {attempt + 1}/{max_retries - 1}...")
                        time.sleep(2)
                    else:
                        print(f"    Failed after {max_retries} attempts: {e}")
                        raise
            
            # Small delay between batches
            time.sleep(0.5)
        
        db.persist()
        print(" All code chunks added successfully")
    else:
        print(" No new code documents to add")


def calculate_chunk_ids(chunks):
    """Generate unique IDs for each chunk"""
    last_source = None
    current_chunk_index = 0
    
    for chunk in chunks:
        source = chunk.metadata.get("source")
        current_source = f"code:{source}"
        
        if current_source == last_source:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        
        chunk_id = f"{current_source}:{current_chunk_index}"
        last_source = current_source
        chunk.metadata["id"] = chunk_id
    
    return chunks


if __name__ == "__main__":
    main()
