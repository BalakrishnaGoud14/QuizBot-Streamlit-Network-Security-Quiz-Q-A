import streamlit as st
import re
import sys
import warnings
from pathlib import Path
from nsrag.rag import answer_question, build_vectorstore

warnings.filterwarnings("ignore")

# Add nsrag to path
sys.path.insert(0, str(Path(__file__).parent / "nsrag"))

from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate
import random

# Page configuration
st.set_page_config(
    page_title="QuizBot - AI Quiz Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (same as before)
st.markdown("""
<style>
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
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Question card styling */
    .question-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 2px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .question-text {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
        line-height: 1.6;
    }
    
    /* Correct answer styling */
    .correct-answer {
        background-color: #d4edda !important;
        border: 2px solid #28a745 !important;
        color: #155724 !important;
        font-weight: 600 !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
        border-radius: 8px !important;
    }
    
    /* Wrong answer styling */
    .wrong-answer {
        background-color: #f8d7da !important;
        border: 2px solid #dc3545 !important;
        color: #721c24 !important;
        font-weight: 600 !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
        border-radius: 8px !important;
    }
    
    /* Neutral option styling */
    .neutral-option {
        background: #f8f9fa;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        color: #495057;
    }
    
    /* Button-style quiz options */
    div[data-testid="column"] button {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%) !important;
        color: #2c3e50 !important;
        border: 2px solid #d0d0d0 !important;
        border-radius: 15px !important;
        padding: 1.25rem !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        transition: all 0.3s !important;
        text-align: left !important;
        height: auto !important;
        min-height: 70px !important;
        white-space: normal !important;
        word-wrap: break-word !important;
    }
    
    div[data-testid="column"] button:hover {
        border-color: #667eea !important;
        background: linear-gradient(135deg, #f0f4ff 0%, #e8eeff 100%) !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.3) !important;
    }
    
    div[data-testid="column"] button:active {
        transform: translateY(0) !important;
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
if 'parsed_questions' not in st.session_state:
    st.session_state.parsed_questions = []
if 'correct_answers' not in st.session_state:
    st.session_state.correct_answers = {}
if 'explanations' not in st.session_state:
    st.session_state.explanations = {}  



# Initialize model
@st.cache_resource
def load_model():
    return Ollama(model="llama3.2:latest")

@st.cache_data
def load_pdf_content():
    """Load PDF content directly"""
    try:
        from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
        loader = PyPDFDirectoryLoader("nsrag/data")
        documents = loader.load()
        return "\n\n---\n\n".join([doc.page_content for doc in documents[:50]])  # Limit for performance
    except:
        # Fallback: use pre-loaded context (silently)
        return """Network security covers cryptography, authentication, protocols, and security mechanisms.
Key topics include: RSA encryption, symmetric/asymmetric encryption, hash functions, digital signatures,
TLS/SSL protocols, key exchange mechanisms, and various attack vectors."""

try:
    model = load_model()
    pdf_content = load_pdf_content()
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

def parse_mcq_questions(text):
    """Parse MCQ questions from text"""
    questions = []
    lines = text.strip().split('\n')
    current_question = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if it's a question line
        if line.startswith('Question') or (line[0].isdigit() and '.' in line[:3]):
            if current_question:
                questions.append(current_question)
            current_question = {
                'question': line.split(':', 1)[-1].strip() if ':' in line else line,
                'options': []
            }
        # Check if it's an option
        elif current_question and line and line[0] in ['A', 'B', 'C', 'D'] and (line[1] == ')' or line[1] == '.'):
            option_text = line[2:].strip()
            current_question['options'].append((line[0], option_text))
    
    if current_question and current_question['options']:
        questions.append(current_question)
    
    return questions

# def parse_tf_questions(text):
#     """Parse True/False questions from text"""
#     questions = []
#     lines = text.strip().split('\n')
    
#     for line in lines:
#         line = line.strip()
#         if not line:
#             continue
            
#         # Check if it's a question line
#         if line.startswith('Question') or (line[0].isdigit() and '.' in line[:3]):
#             question_text = line.split(':', 1)[-1].strip() if ':' in line else line
#             # Remove question number if present
#             if question_text[0].isdigit():
#                 question_text = question_text.split('.', 1)[-1].strip()
#             questions.append({
#                 'question': question_text,
#                 'options': [('True', 'True'), ('False', 'False')]
#             })
    
#     return questions

def parse_tf_questions(text):
    """Parse True/False questions from text and force clean options"""
    questions = []
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Identify question line
        if line.lower().startswith('question') or (line[0].isdigit() and '.' in line[:3]):
            question_text = line.split(':', 1)[-1].strip() if ':' in line else line
            # Remove leftover "True or False:" or other extra formatting
            question_text = question_text.replace('True or False:', '').replace('True)True', '').replace('False)False', '').strip()
            
            questions.append({
                'question': question_text,
                'options': [('True','True'), ('False','False')]  # Force clean options
            })
    
    return questions


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
    <h1>QuizBot</h1>
    <p>AI-Powered Network Security Quiz Generator</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio(
        "Choose a mode:",
        ["Generate Quiz", "Ask Questions", "About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### Settings")
    
    if page == "Generate Quiz":
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
    st.markdown("### Statistics")
    st.metric("Topics Available", len(TOPICS))
    st.metric("Model", "llama3.2")

# Main content
if page == "Generate Quiz":
    st.markdown("## Quiz Generator")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("Generate New Quiz", use_container_width=True):
            with st.spinner("Generating quiz..."):
                try:
                    # Create prompt based on settings
                    if st.session_state.topic_mode == "Random Topics":
                        topics = random.sample(TOPICS, 2)
                        topic_text = f"on the topics: {', '.join(topics)}"
                    else:
                        topic_text = f"on the topic: {st.session_state.selected_topic}"
                    
                    if st.session_state.quiz_type == "Multiple Choice (MCQ)":
                        prompt = f"""Based on network security concepts, generate {st.session_state.num_questions} multiple-choice questions {topic_text}.

Each question should have:
- A clear question
- 4 options labeled A, B, C, D
- DO NOT show the correct answer in the output

Format:
Question 1: [question text]
A) [option]
B) [option]
C) [option]
D) [option]

Generate the quiz now (without showing correct answers):"""
                    else:
                        prompt = f"""Based on network security concepts, generate {st.session_state.num_questions} true/false questions {topic_text}.

Requirements:
1. Each question should start with 'Question X:'.
2. Do NOT include 'True or False:' in the question.
3. Do NOT include letters, parentheses, or repeated words.
4. The options line should be exactly: True    False
5. Do NOT show the correct answer.

Format:
Question 1: [statement]
Options: True    False

Question 2: [statement]
Options: True    False

Generate the quiz now:"""

#                         prompt = f"""Based on network security concepts, generate {st.session_state.num_questions} true/false questions {topic_text}.

# Format:
# Question 1: [statement]

# DO NOT show the answers. Generate the quiz now:"""
                    
                    # Use PDF content as context
                    full_prompt = f"Context from network security materials:\n{pdf_content[:3000]}\n\n{prompt}"
                    
                    response = model.invoke(full_prompt)
                    
                    # Parse questions
                    if st.session_state.quiz_type == "Multiple Choice (MCQ)":
                        parsed = parse_mcq_questions(response)
                    else:
                        parsed = parse_tf_questions(response)
                    
                    st.session_state.quiz_data = {
                        'questions': response,
                        'type': st.session_state.quiz_type
                    }
                    st.session_state.parsed_questions = parsed
                    st.session_state.user_answers = {}
                    st.session_state.quiz_submitted = False
                    st.session_state.correct_answers = {}
                    
                    st.success("Quiz generated successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with col2:
        if st.session_state.quiz_data and not st.session_state.quiz_submitted:
            if st.button("Submit Quiz", use_container_width=True):
                st.session_state.quiz_submitted = True
                st.rerun()
    
    # Display quiz
    if st.session_state.quiz_data and st.session_state.get('parsed_questions'):
        st.markdown("---")
        
        if not st.session_state.quiz_submitted:
            st.markdown("### Your Quiz")
            
            # Display each question with button-style options
            for idx, q in enumerate(st.session_state.parsed_questions):
                st.markdown(f"""
                <div class="question-card">
                    <div class="question-text">Question {idx + 1}: {q['question']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                if q['options']:
                    # Create columns for button-style options
                    cols = st.columns(len(q['options']))
                    for col_idx, (opt_letter, opt_text) in enumerate(q['options']):
                        with cols[col_idx]:
                            if st.button(
                                f"{opt_text}",
                                key=f"q_{idx}_opt_{opt_letter}",
                                use_container_width=True
                            ):
                                st.session_state.user_answers[idx] = opt_letter
                    
                    # Show selected answer
                    if idx in st.session_state.user_answers:
                        st.info(f"Selected: {st.session_state.user_answers[idx]}")
                
                st.markdown("<br>", unsafe_allow_html=True)
        
        else:
            # Get correct answers first
            # Evaluate only if not already evaluated
            if not st.session_state.correct_answers:
                with st.spinner("Evaluating your answers..."):
                    try:
                        user_ans_str = ", ".join([
                            f"{i+1}. {st.session_state.user_answers.get(i, 'No answer')}" 
                            for i in range(len(st.session_state.parsed_questions))
                        ])
            
                        eval_prompt = f"""You are a quiz evaluator. 

        Here are the quiz questions:
        {st.session_state.quiz_data['questions']}

        The user's answers are:
        {user_ans_str}

        For each question, provide ONLY:
        1. The correct answer (just the letter for MCQ or True/False)
        2. A brief explanation (one sentence)

        Format your response as:
        Question 1: Correct Answer: [X], Explanation: [brief explanation]
        Question 2: Correct Answer: [X], Explanation: [brief explanation]
        etc."""
                    
                        evaluation = model.invoke(eval_prompt)

                        # Regex to parse "Question #: Correct Answer: [X], Explanation: ..."
                        pattern = r'Question\s*\d+:\s*Correct Answer:\s*\[?([A-DTFa-dtf]+)\]?\s*,?\s*Explanation:\s*(.+)'
                        
                        for match in re.finditer(pattern, evaluation):
                            correct_ans = match.group(1).strip().upper()
                            explanation = match.group(2).strip()
                            
                            # Determine question index by order
                            idx = len(st.session_state.correct_answers)
                            
                            st.session_state.correct_answers[idx] = correct_ans
                            st.session_state.explanations[idx] = explanation
                            
                    except Exception as e:
                        st.error(f"Error evaluating quiz: {e}")
#             if not st.session_state.correct_answers:
#                 with st.spinner("Evaluating your answers..."):
#                     try:
#                         # Build answer string
#                         user_ans_str = ", ".join([f"{i+1}. {st.session_state.user_answers.get(i, 'No answer')}" 
#                                                   for i in range(len(st.session_state.parsed_questions))])
                        
#                         eval_prompt = f"""You are a quiz evaluator. 

# Here are the quiz questions:
# {st.session_state.quiz_data['questions']}

# The user's answers are: 
# {user_ans_str}

# For each question, provide ONLY:
# 1. The correct answer (just the letter for MCQ or True/False)
# 2. A brief explanation (one sentence)

# Format your response as:
# Question 1: Correct Answer: [X], Explanation: [brief explanation]
# Question 2: Correct Answer: [X], Explanation: [brief explanation]
# etc."""
                        
#                         evaluation = model.invoke(eval_prompt)
                        
#                         # Parse correct answers
#                         for idx, line in enumerate(evaluation.split('\n')):
#                             if 'Correct Answer:' in line:
#                                 try:
#                                     correct = line.split('Correct Answer:')[1].split(',')[0].strip()
#                                     # Extract just the letter/answer
#                                     correct = correct.replace('[', '').replace(']', '').strip()
#                                     if correct and correct[0] in ['A', 'B', 'C', 'D', 'T', 'F']:
#                                         st.session_state.correct_answers[idx] = correct[0]
#                                 except:
#                                     pass
#                     except Exception as e:
#                         st.error(f"Error: {e}")
            
            # Display results with color coding
            st.markdown("### Quiz Results")

            correct_count = 0

            for idx, q in enumerate(st.session_state.parsed_questions):
                user_answer = st.session_state.user_answers.get(idx, 'No answer')
                correct_answer = st.session_state.correct_answers.get(idx, '?')

                # Normalize for both MCQ and True/False
                user_answer_norm = str(user_answer).strip().lower()
                correct_answer_norm = str(correct_answer).strip().lower()

                is_correct = user_answer_norm == correct_answer_norm
                if is_correct:
                    correct_count += 1

                # Display question
                st.markdown(f"""
                <div class="question-card">
                    <div class="question-text">Question {idx + 1}: {q['question']}</div>
                </div>
                """, unsafe_allow_html=True)

                # Display options
                for opt_letter, opt_text in q['options']:

                    opt_letter_norm = str(opt_letter).strip().lower()

                    is_user_choice = opt_letter_norm == user_answer_norm
                    is_correct_answer = opt_letter_norm == correct_answer_norm

                    if is_correct_answer:
                        style_class = "correct-answer"
                        indicator = " ✓ Correct Answer"
                    elif is_user_choice and not is_correct:
                        style_class = "wrong-answer"
                        indicator = " ✗ Your Answer"
                    else:
                        style_class = "neutral-option"
                        indicator = ""

                    st.markdown(f"""
                    <div class="{style_class}">
                        {opt_text} {indicator}
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

            # st.markdown("### Quiz Results")
            
            # correct_count = 0
            # for idx, q in enumerate(st.session_state.parsed_questions):
            #     user_answer = st.session_state.user_answers.get(idx, 'No answer')
            #     correct_answer = st.session_state.correct_answers.get(idx, '?')
            #     is_correct = user_answer == correct_answer
                
            #     if is_correct:
            #         correct_count += 1
                
            #     # Display question
            #     st.markdown(f"""
            #     <div class="question-card">
            #         <div class="question-text">Question {idx + 1}: {q['question']}</div>
            #     </div>
            #     """, unsafe_allow_html=True)
                
            #     # Display options with color coding
            #     for opt_letter, opt_text in q['options']:
            #         is_user_choice = opt_letter == user_answer
            #         is_correct_answer = opt_letter == correct_answer
                    
            #         if is_correct_answer:
            #             style_class = "correct-answer"
            #             indicator = " ✓ Correct Answer"
            #         elif is_user_choice and not is_correct:
            #             style_class = "wrong-answer"
            #             indicator = " ✗ Your Answer"
            #         else:
            #             style_class = "neutral-option"
            #             indicator = ""
                    
            #         st.markdown(f"""
            #         <div class="{style_class}">
            #             {opt_text} {indicator}
            #         </div>
            #         """, unsafe_allow_html=True)
                
            #     st.markdown("<br>", unsafe_allow_html=True)
            
            # Show score
            total = len(st.session_state.parsed_questions)
            percentage = (correct_count / total * 100) if total > 0 else 0
            
            st.markdown("---")
            st.markdown(f"### Score: {correct_count}/{total} ({percentage:.1f}%)")
            
            if st.button("Generate New Quiz"):
                st.session_state.quiz_data = None
                st.session_state.parsed_questions = []
                st.session_state.user_answers = {}
                st.session_state.quiz_submitted = False
                st.session_state.correct_answers = {}
                st.rerun()

elif page == "Ask Questions":
    st.markdown("## Ask Questions")
    
    # Chat history
    for question, answer in st.session_state.chat_history:
        with st.container():
            st.markdown(f"**You:** {question}")
            st.markdown(f"**QuizBot:** {answer}")
            st.markdown("---")
    
    # Question input
    with st.form("question_form", clear_on_submit=True):
        question = st.text_input("Your question:", placeholder="e.g., What is RSA encryption?")
        submitted = st.form_submit_button("Ask", use_container_width=True)
        
        if submitted and question:
            with st.spinner("Thinking..."):
                try:
                    answer, sources = answer_question(question, use_web=True)
                except Exception as e:
                    st.error(f" Error: {e}")
                    st.stop()
                try:
                    prompt = f"""Based on network security concepts, answer this question:

Question: {question}

Context from materials:
{pdf_content[:2000]}

Provide a clear, detailed answer:"""
                    prompt = f"""You are an expert assistant. Answer the user’s question clearly and concisely. 
Do NOT include any reference links or sources inside the answer text. 
After providing the answer, list all sources under a separate section titled 'References'.

Format:

<Your answer here, clean and free of inline references> {answer}

References:
- <source 1>
- <source 2>
- <source 3>
{sources}"""
#                     prompt = f"""
# You are an expert assistant. Answer the user’s question clearly and concisely. 

# Rules:
# 1. Provide a clear answer based on the context (PDF content or knowledge you have). 
# 2. Do NOT include reference links inside the answer text.
# 3. If there is no relevant information in the context, provide a concise answer based on general knowledge.
# 4. In the 'References' section, always include relevant links:
#    - Strictly use only the sources provided in the context or RAG system if available.
#    - If no sources are available, include reputable external references, preferably Wikipedia or official sites.

# Format:

# Answer:
# <Your answer here, clean and free of inline references> {answer}

# References:
#  - <source 1>
#  - <source 2>
#  - <source 3>
#  if {sources} else '- https://en.wikipedia.org/wiki/' """

                    answer = model.invoke(prompt)
                    st.session_state.chat_history.append((question, answer))
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {e}")

else:  # About page
    st.markdown("## About QuizBot")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Topics", len(TOPICS))
    with col2:
        st.metric("Model", "llama3.2")
    with col3:
        st.metric("Interfaces", "5")
    
    st.markdown("---")
    
    st.markdown("""
    ### Features
    
    - **Quiz Generation**: MCQ and True/False quizzes
    - **Q&A System**: Ask questions and get detailed answers
    - **AI-Powered**: Uses llama3.2 for accurate responses
    - **100% Local**: All processing on your machine
    
    ### Technology
    
    - **LLM**: Ollama (llama3.2)
    - **UI**: Streamlit
    - **Framework**: LangChain
    - **Backend**: Python
    
    ### Getting Started
    
    1. Select "Generate Quiz" to create a quiz
    2. Choose your preferences in the sidebar
    3. Click "Generate New Quiz"
    4. Answer the questions and submit!
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; padding: 1rem;">
    <p>QuizBot - Powered by llama3.2 | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
