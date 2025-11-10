
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate

# Initialize the model
model = Ollama(model="llama3.2:latest")

CODE_PATH = ".."
CODE_EXTENSIONS = ['.py', '.js', '.jsx', '.html', '.css', '.json', '.md']

def load_codebase():
    """Load all relevant code files"""
    code_files = {}
    code_path = Path(CODE_PATH)
    
    skip_dirs = {'node_modules', '__pycache__', '.git', 'build', 'dist', 'venv', 'chroma', 'data'}
    
    for file_path in code_path.rglob('*'):
        if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
            continue
            
        if file_path.suffix in CODE_EXTENSIONS and file_path.is_file():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    relative_path = str(file_path.relative_to(code_path))
                    code_files[relative_path] = f.read()
            except:
                pass
    
    return code_files


def find_relevant_files(query: str, code_files: dict, max_files: int = 5):
    """Find files most relevant to the query based on keywords"""
    query_lower = query.lower()
    scored_files = []
    
    for file_path, content in code_files.items():
        score = 0
        content_lower = content.lower()
        
        # Score based on query words appearing in file
        for word in query_lower.split():
            if len(word) > 3:  # Only consider meaningful words
                score += content_lower.count(word)
        
        # Boost score for files mentioned in query
        if any(part in query_lower for part in file_path.lower().split('/')):
            score += 10
        
        if score > 0:
            scored_files.append((file_path, content, score))
    
    # Sort by score and return top files
    scored_files.sort(key=lambda x: x[2], reverse=True)
    return scored_files[:max_files]


def ask_about_code(question: str):
    """Ask a question about the codebase"""
    print(f"\n Question: {question}")
    print(" Loading codebase...")
    
    code_files = load_codebase()
    print(f" Loaded {len(code_files)} files")
    
    print(" Finding relevant files...")
    relevant_files = find_relevant_files(question, code_files)
    
    if not relevant_files:
        print(" No relevant files found")
        return
    
    print(f" Found {len(relevant_files)} relevant files:")
    for file_path, _, score in relevant_files:
        print(f"  - {file_path} (relevance: {score})")
    
    # Build context from relevant files
    context = "\n\n".join([
        f"File: {file_path}\n```\n{content[:2000]}{'...' if len(content) > 2000 else ''}\n```"
        for file_path, content, _ in relevant_files
    ])
    
    # Create prompt
    prompt_template = ChatPromptTemplate.from_template("""
You are a helpful code assistant analyzing the QuizBot project.

Here are the relevant code files:

{context}

---

Question: {question}

Please provide a clear and detailed answer based on the code above.
""")
    
    prompt = prompt_template.format(context=context, question=question)
    
    print("\n Thinking...")
    response = model.invoke(prompt)
    
    print("\n Answer:")
    print(response)
    print("\n" + "="*60)
    
    return response


def interactive_mode():
    """Run in interactive mode"""
    print("="*60)
    print(" QuizBot Code Assistant (powered by llama3.2)")
    print("="*60)
    print("\nAsk questions about the QuizBot codebase!")
    print("Type 'exit' or 'quit' to stop\n")
    
    while True:
        try:
            question = input(" Your question: ").strip()
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("\n Goodbye!")
                break
            
            if not question:
                continue
            
            ask_about_code(question)
            
        except KeyboardInterrupt:
            print("\n\n Goodbye!")
            break
        except Exception as e:
            print(f"\n Error: {e}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Single question mode
        question = " ".join(sys.argv[1:])
        ask_about_code(question)
    else:
        # Interactive mode
        interactive_mode()
