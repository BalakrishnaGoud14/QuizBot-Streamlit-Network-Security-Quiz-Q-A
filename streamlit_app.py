
"""
QuizBot -  Streamlit Interface
"""
import streamlit as st
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Add nsrag to path
sys.path.insert(0, str(Path(__file__).parent / "nsrag"))

from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
import random

# Import from nsrag directory
try:
    from nsrag.get_embedding_function import get_embedding_function
    from nsrag.prompt_templates import *
except ImportError:
    # If running from nsrag directory
    from get_embedding_function import get_embedding_function
    from prompt_templates import *

# Page configuration
st.set_page_config(
    page_title="QuizBot - AI Quiz Generator",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #4F46E5;
        --secondary-color: #7C3AED;
        --success-color: #10B981;
        --error-color: #EF4444;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        font-size: 3rem;
        margin: 0;
        font-weight: 700;
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Card styling */
    .stCard {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Question styling */
    .question-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid #4F46E5;
    }
    
    .question-text {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1F2937;
        margin-bottom: 1rem;
    }
    
    .option-item {
        background: white;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border: 2px solid #E5E7EB;
        transition: all 0.3s;
    }
    
    .option-item:hover {
        border-color: #4F46E5;
        transform: translateX(5px);
    }
    
    /* Correct/Wrong answer styling */
    .correct-answer {
        background: #D1FAE5;
        border-color: #10B981;
        color: #065F46;
        font-weight: 600;
    }
    
    .wrong-answer {
        background: #FEE2E2;
        border-color: #EF4444;
        color: #991B1B;
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .stat-label {
        font-size: 1rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Success/Error messages */
    .success-message {
        background: #D1FAE5;
        color: #065F46;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #10B981;
    }
    
    .error-message {
        background: #FEE2E2;
        color: #991B1B;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #EF4444;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'quiz_data' not in st.session_state:
    st.session_state.quiz_data = None
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = {}
if 'quiz_submitted' not in st.session_state:
    st.session_state.quiz_submitted = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize models
@st.cache_resource
def load_models():
    import chromadb
    
    model = Ollama(model="llama3.2:latest")
    embedding_function = get_embedding_function()
    
    # Determine correct path to chroma directory
    chroma_path = Path("nsrag/chroma")
    if not chroma_path.exists():
        chroma_path = Path("chroma")
    
    # Ensure the directory exists
    chroma_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Try the new PersistentClient API
        client = chromadb.PersistentClient(path=str(chroma_path))
        
        # List existing collections
        collections = client.list_collections()
        
        if collections:
            # Use the first collection found
            collection_name = collections[0].name
        else:
            # Create a new collection if none exist
            collection_name = "langchain"
            client.create_collection(name=collection_name)
        
        db = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding_function
        )
    except Exception as e:
        # Fallback to old API
        st.warning(f"Using fallback initialization method: {e}")
        db = Chroma(
            persist_directory=str(chroma_path),
            embedding_function=embedding_function
        )
    
    return model, db

try:
    model, db = load_models()
    st.success(" Models loaded successfully!")
except Exception as e:
    st.error(f" Error loading models: {e}")
    st.warning("""
    **Troubleshooting Steps:**
    1. Make sure Ollama is running: `pgrep -fl ollama`
    2. Check if models are installed: `ollama list`
    3. Verify the database exists: `ls -la nsrag/chroma/`
    4. Try rebuilding the database: `cd nsrag && python3 populate_database.py --reset`
    """)
    
    if st.button(" Retry Loading Models"):
        st.cache_resource.clear()
        st.rerun()
    
    st.stop()

# Topics list
TOPICS = [
    "OSI architecture", "Symmetric Encryption", "Rijndael", "Entropy",
    "Pseudorandom Number Generator", "Block and Stream Ciphers", "RC4 Stream Cipher",
    "Public-Key Cryptography", "RSA", "Homomorphic encryption",
    "Message authentication", "Hash functions", "Secure Hash Function",
    "Length Extension Attacks", "Message Authentication Code", "HMAC",
    "Authenticated Encryption", "TLS 1.0 Lucky 13 Attack", "Digital Signatures",
    "Hybrid Encryption", "Symmetric key distribution", "Diffie-Hellman Key Exchange"
]

# Header
st.markdown("""
<div class="main-header">
    <h1>🎓 QuizBot</h1>
    <p>AI-Powered Network Security Quiz Generator</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("###  Navigation")
    page = st.radio(
        "Choose a mode:",
        [" Generate Quiz", " Ask Questions", " Code Assistant", " About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("###  Settings")
    
    if page == " Generate Quiz":
        quiz_type = st.selectbox(
            "Quiz Type",
            ["Multiple Choice (MCQ)", "True/False"],
            key="quiz_type"
        )
        
        topic_mode = st.radio(
            "Topic Selection",
            ["Random Topics", "Specific Topic"],
            key="topic_mode"
        )
        
        if topic_mode == "Specific Topic":
            selected_topic = st.selectbox("Choose Topic", TOPICS, key="selected_topic")
        
        num_questions = st.slider("Number of Questions", 3, 10, 5, key="num_questions")
    
    st.markdown("---")
    st.markdown("###  Stats")
    st.metric("Documents Indexed", "281")
    st.metric("Topics Available", len(TOPICS))
    st.metric("Model", "llama3.2")

# Main content area
if page == " Generate Quiz":
    st.markdown("##  Quiz Generator")
    st.markdown("Generate AI-powered quizzes on network security topics!")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button(" Generate New Quiz", use_container_width=True):
            with st.spinner(" Generating quiz questions..."):
                try:
                    # Determine query based on settings
                    if st.session_state.topic_mode == "Random Topics":
                        selected_topics = random.sample(TOPICS, 2)
                        query = f"Give me information about {', '.join(selected_topics)}"
                    else:
                        query = f"Give me information about {st.session_state.selected_topic}"
                    
                    # Get relevant documents
                    results = db.similarity_search_with_score(query, k=7)
                    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
                    
                    # Generate quiz
                    if st.session_state.quiz_type == "Multiple Choice (MCQ)":
                        prompt_template = ChatPromptTemplate.from_template(QUIZ_MCQ_GENERAL_PROMPT)
                    else:
                        prompt_template = ChatPromptTemplate.from_template(QUIZ_TF_GENERAL_PROMPT)
                    
                    prompt = prompt_template.format(context=context_text)
                    response_text = model.invoke(prompt)
                    
                    # Store quiz data
                    st.session_state.quiz_data = {
                        'questions': response_text,
                        'context': context_text,
                        'type': st.session_state.quiz_type
                    }
                    st.session_state.user_answers = {}
                    st.session_state.quiz_submitted = False
                    
                    st.success(" Quiz generated successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f" Error generating quiz: {e}")
    
    with col2:
        if st.session_state.quiz_data and not st.session_state.quiz_submitted:
            if st.button(" Submit Quiz", use_container_width=True):
                st.session_state.quiz_submitted = True
                st.rerun()
    
    # Display quiz
    if st.session_state.quiz_data:
        st.markdown("---")
        
        if not st.session_state.quiz_submitted:
            st.markdown("###  Your Quiz")
            st.markdown(st.session_state.quiz_data['questions'])
            
            st.markdown("###  Your Answers")
            answer_input = st.text_area(
                "Enter your answers (e.g., 1. A 2. B 3. C or 1. True 2. False 3. True)",
                height=100,
                key="answer_input"
            )
            st.session_state.user_answers = answer_input
            
        else:
            # Evaluate quiz
            st.markdown("###  Quiz Results")
            
            with st.spinner(" Evaluating your answers..."):
                try:
                    if st.session_state.quiz_data['type'] == "Multiple Choice (MCQ)":
                        eval_template = ChatPromptTemplate.from_template(EVAL_QUIZ_MCQ_GENERAL_PROMPT)
                    else:
                        eval_template = ChatPromptTemplate.from_template(EVAL_QUIZ_TF_GENERAL_PROMPT)
                    
                    eval_prompt = eval_template.format(
                        context=st.session_state.quiz_data['context'],
                        questions=st.session_state.quiz_data['questions'],
                        usrAns=st.session_state.user_answers
                    )
                    
                    evaluation = model.invoke(eval_prompt)
                    
                    st.markdown("####  Evaluation")
                    st.markdown(evaluation)
                    
                    if st.button(" Generate New Quiz"):
                        st.session_state.quiz_data = None
                        st.session_state.user_answers = {}
                        st.session_state.quiz_submitted = False
                        st.rerun()
                        
                except Exception as e:
                    st.error(f" Error evaluating quiz: {e}")

elif page == " Ask Questions":
    st.markdown("##  Ask Questions")
    st.markdown("Ask any question about network security topics!")
    
    # Chat interface
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"** You:** {question}")
            st.markdown(f"** QuizBot:** {answer}")
            st.markdown("---")
    
    # Question input
    with st.form("question_form", clear_on_submit=True):
        question = st.text_input("Your question:", placeholder="e.g., What is RSA encryption?")
        submitted = st.form_submit_button(" Ask", use_container_width=True)
        
        if submitted and question:
            with st.spinner(" Thinking..."):
                try:
                    # Get relevant documents
                    results = db.similarity_search_with_score(question, k=7)
                    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
                    
                    # Generate answer
                    prompt_template = ChatPromptTemplate.from_template(OPEN_ENDED_QUESTION_PROMPT)
                    prompt = prompt_template.format(context=context_text, question=question)
                    answer = model.invoke(prompt)
                    
                    # Add to history
                    st.session_state.chat_history.append((question, answer))
                    st.rerun()
                    
                except Exception as e:
                    st.error(f" Error: {e}")

elif page == " Code Assistant":
    st.markdown("##  Code Assistant")
    st.markdown("Ask questions about the QuizBot codebase!")
    
    # Load code files
    @st.cache_data
    def load_codebase():
        code_files = {}
        code_path = Path(".")
        skip_dirs = {'node_modules', '__pycache__', '.git', 'build', 'dist', 'venv', 'chroma', 'data'}
        
        for file_path in code_path.rglob('*'):
            if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                continue
            if file_path.suffix in ['.py', '.js', '.jsx', '.html', '.css', '.json', '.md'] and file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code_files[str(file_path)] = f.read()
                except:
                    pass
        return code_files
    
    code_files = load_codebase()
    st.info(f" Loaded {len(code_files)} code files")
    
    # Code question input
    code_question = st.text_input("Ask about the code:", placeholder="e.g., How does the quiz generation work?")
    
    if st.button(" Analyze Code", use_container_width=True):
        if code_question:
            with st.spinner(" Analyzing codebase..."):
                try:
                    # Find relevant files
                    query_lower = code_question.lower()
                    scored_files = []
                    
                    for file_path, content in code_files.items():
                        score = sum(content.lower().count(word) for word in query_lower.split() if len(word) > 3)
                        if score > 0:
                            scored_files.append((file_path, content, score))
                    
                    scored_files.sort(key=lambda x: x[2], reverse=True)
                    relevant_files = scored_files[:3]
                    
                    if relevant_files:
                        st.markdown("####  Relevant Files:")
                        for file_path, _, score in relevant_files:
                            st.markdown(f"- `{file_path}` (relevance: {score})")
                        
                        # Build context
                        context = "\n\n".join([
                            f"File: {file_path}\n```\n{content[:1500]}{'...' if len(content) > 1500 else ''}\n```"
                            for file_path, content, _ in relevant_files
                        ])
                        
                        # Generate answer
                        prompt_template = ChatPromptTemplate.from_template("""
You are a helpful code assistant analyzing the QuizBot project.

Here are the relevant code files:

{context}

---

Question: {question}

Please provide a clear and detailed answer based on the code above.
""")
                        prompt = prompt_template.format(context=context, question=code_question)
                        answer = model.invoke(prompt)
                        
                        st.markdown("####  Answer:")
                        st.markdown(answer)
                    else:
                        st.warning("No relevant files found for your question.")
                        
                except Exception as e:
                    st.error(f" Error: {e}")

else:  # About page
    st.markdown("##  About QuizBot")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <p class="stat-number">281</p>
            <p class="stat-label">Documents</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <p class="stat-number">22</p>
            <p class="stat-label">Topics</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
            <p class="stat-number">30</p>
            <p class="stat-label">PDF Files</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    ###  Features
    
    - ** Quiz Generation**: Generate MCQ and True/False quizzes on network security topics
    - ** Q&A System**: Ask open-ended questions and get detailed answers
    - ** Code Assistant**: Understand how the QuizBot codebase works
    - ** AI-Powered**: Uses llama3.2 with RAG for accurate, context-aware responses
    
    ###  Technology Stack
    
    - **LLM**: Ollama (llama3.2:latest)
    - **Embeddings**: nomic-embed-text
    - **Vector DB**: ChromaDB
    - **Framework**: LangChain
    - **UI**: Streamlit
    - **Backend**: Python, Flask
    
    ###  Topics Covered
    
    Network Security fundamentals including:
    - Cryptography (Symmetric & Asymmetric)
    - Hash Functions & Message Authentication
    - Digital Signatures & Certificates
    - Network Protocols & Security
    - And much more!
    
    ###  Getting Started
    
    1. Select a mode from the sidebar
    2. Generate quizzes or ask questions
    3. Learn and improve your network security knowledge!
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; padding: 1rem;">
    <p> QuizBot - Powered by llama3.2 & LangChain | Made with  using Streamlit</p>
</div>
""", unsafe_allow_html=True)
