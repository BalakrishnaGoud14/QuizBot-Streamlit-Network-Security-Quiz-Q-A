import warnings
warnings.filterwarnings("ignore")
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function

from prompt_templates import *

import random

#test
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_core.prompts import ChatPromptTemplate

# Load PDFs directly from your local folder
DATA_PATH = "data"

def query_rag(query_text: str):
    # Step 1: Load PDFs
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()

    # Step 2: Combine all text 
    context_text = "\n\n---\n\n".join([doc.page_content for doc in documents])
    #print(context_text)
    question=query_text
    context=context_text
    # Step 3: Create the prompt
    prompt_template = ChatPromptTemplate.from_template("""
    You are a helpful tutor. Use the following context to answer the question.

    Context:
    {context}

    Question: {question}

    Answer:
    """)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Step 4: Get answer from your LLM
    response = model.invoke(prompt)
    print(response)



# Global list of topics
TOPICS = [
    "OSI architecture",
    "Symmetric Encryption",
    "Rijndael",
    "Entropy",
    "Pseudorandom Number Generator",
    "Block and Stream Ciphers",
    "RC4 Stream Cipher",
    "Public-Key Cryptography",
    "RSA",
    "Attack approaches - page 159(merged pdf)",
    "Homomorphic encryption",
    "Message authentication - page 166(merged pdf)",
    "Hash functions and Requirements for secure hash functions",
    "Secure Hash Function",
    "Length Extension Attacks",
    "Message Authentication Code",
    "HMAC",
    "Authenticated Encryption",
    "TLS 1.0 “Lucky 13” Attack",
    "Digital Signatures",
    "Hybrid Encryption",
    "symmetric key distribution",
    "Diffie-Hellman Key Exchange"
]

CHROMA_PATH = "chroma"

model = Ollama(model="llama3.2:latest")
embedding_function = get_embedding_function()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

"""def query_rag(query_text: str):
    results = db.similarity_search_with_score(query_text, k=7)
    #test
    '''
    for doc, score in results:
        print(doc.metadata["source"], score)
        print(doc.page_content[:200], "\n---\n")
    '''
    #test
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(OPEN_ENDED_QUESTION_PROMPT)
    prompt = prompt_template.format(context=context_text, question=query_text)
    response_text = model.invoke(prompt)
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text
"""
    
def openQs(que):
    query_rag(que)

def quizMCQ():
    print("Generating a general MCQ Quiz: ")
    # Randomly select two topics
    selected_topics = random.sample(TOPICS, 2)
    dbQuery = f"Give me a set of multiple-choice questions on {', '.join(selected_topics)}."
    results = db.similarity_search_with_score(dbQuery, k=7)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    prompt_template = ChatPromptTemplate.from_template(QUIZ_MCQ_GENERAL_PROMPT)
    prompt = prompt_template.format(context=context_text)
    response_text = model.invoke(prompt)
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

    askAns = '''\nGive your answers for each question like this: 1. A 2. B 3. C 4. D 5. A\n'''
    usrAns = input(askAns)
    eval_prompt_template = ChatPromptTemplate.from_template(EVAL_QUIZ_MCQ_GENERAL_PROMPT)
    eval_prompt = eval_prompt_template.format(context=context_text, questions=formatted_response, usrAns=usrAns)
    response_text = model.invoke(eval_prompt)
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


def quizTF():
    print("Generating a general T/F Quiz: ")
    # Randomly select two topics
    selected_topics = random.sample(TOPICS, 2)
    dbQuery = f"Give me a set of true/false questions on {', '.join(selected_topics)}."
    results = db.similarity_search_with_score(dbQuery, k=7)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    prompt_template = ChatPromptTemplate.from_template(QUIZ_TF_GENERAL_PROMPT)
    prompt = prompt_template.format(context=context_text)
    response_text = model.invoke(prompt)
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

    askAns = '''\nGive your answers for each question like this: 1. True 2. False 3. True 4. False 5. True\n'''
    usrAns = input(askAns)
    eval_prompt_template = ChatPromptTemplate.from_template(EVAL_QUIZ_TF_GENERAL_PROMPT)
    eval_prompt = eval_prompt_template.format(context=context_text, questions=formatted_response, usrAns=usrAns)
    response_text = model.invoke(eval_prompt)
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


def quizMCQByTopic(topic):
    print("Generating MCQ Quiz on " + topic)
    dbQuery = "Give me information about " + topic
    results = db.similarity_search_with_score(dbQuery, k=7)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    prompt_template = ChatPromptTemplate.from_template(QUIZ_MCQ_TOPIC_PROMPT)
    prompt = prompt_template.format(context=context_text)
    response_text = model.invoke(prompt)
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

    askAns = '''\nGive your answers for each question like this: 1. A 2. B 3. C 4. D 5. A\n'''
    usrAns = input(askAns)
    eval_prompt_template = ChatPromptTemplate.from_template(EVAL_QUIZ_MCQ_TOPIC_PROMPT)
    eval_prompt = eval_prompt_template.format(context=context_text, questions=formatted_response, usrAns=usrAns)
    response_text = model.invoke(eval_prompt)
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

def quizTFByTopic(topic):
    print("Generating T/F Quiz on " + topic)
    dbQuery = "Give me information about " + topic
    results = db.similarity_search_with_score(dbQuery, k=7)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    prompt_template = ChatPromptTemplate.from_template(QUIZ_TF_TOPIC_PROMPT)
    prompt = prompt_template.format(context=context_text)
    response_text = model.invoke(prompt)
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

    askAns = '''\nGive your answers for each question like this: 1. True 2. False 3. True 4. False 5. True\n'''
    usrAns = input(askAns)
    eval_prompt_template = ChatPromptTemplate.from_template(EVAL_QUIZ_TF_TOPIC_PROMPT)
    eval_prompt = eval_prompt_template.format(context=context_text, questions=formatted_response, usrAns=usrAns)
    response_text = model.invoke(eval_prompt)
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    while True:
        menu = '''
        1. Open ended questions\n 
        2. Quiz MCQ\n 
        3. Quiz MCQ by topic\n 
        4. Quiz T/F\n 
        5. Quiz T/F by topic\n 
        6. Exit
        '''
        print(menu)
        option = int(input("Select a mode: "))
        if option == 6:
            print("Bye")
            break
        elif option == 1:
            que = input("What's your question?: ")
            openQs(que)
        elif option == 2:
            quizMCQ()
        elif option == 3:
            topic = input("On which topic?: ")
            quizMCQByTopic(topic)
        elif option == 4:
            quizTF()
        elif option == 5:
            topic = input("On which topic?: ")
            quizTFByTopic(topic)
