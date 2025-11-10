from openai import OpenAI
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel
import instructor
from langchain_community.vectorstores import Chroma
from get_embedding_function import get_embedding_function

app = Flask(__name__)
CORS(app)

# enables `response_model` in create call
client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # required, but unused
    ),
    mode=instructor.Mode.JSON,
)

class MCQ(BaseModel):
    question_text: str
    options: list[str]
    correct_answer: str

class MCQArr(BaseModel):
    mcqs: list[MCQ]

CHROMA_PATH = "chroma"
embedding_function = get_embedding_function()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

sysIntr = '''
You are a quiz master and you only output in JSON. 
You are very good at generating multiple-choice quiz questions.
Your task is to generate an array of JSON objects where each object represents a question.
Each question object should have the following structure:
- "question_text": A string representing the question.
- "options": An array of exactly 4 string options for the question.
- "correct_answer": A string that is one of the options from the array and is the correct answer for the question.

The response should strictly follow this JSON structure:
[
  {
    "question_text": "What is the capital of France?",
    "options": ["Berlin", "Madrid", "Paris", "Rome"],
    "correct_answer": "Paris"
  },
  {
    "question_text": "Which planet is known as the Red Planet?",
    "options": ["Earth", "Mars", "Jupiter", "Venus"],
    "correct_answer": "Mars"
  }
]

Generate quiz questions based on the provided context data, ensuring accuracy and diversity.
'''

import json

@app.route('/generate_quiz', methods=['GET'])
def generateQuiz():
    try:
        
        results = db.similarity_search_with_score("Explain Homomorphic encryption", k=7)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        sources = [doc.metadata.get("id", None) for doc, _score in results]
        usrPrompt = f'''
        Perform the following task based only on the following context:

        {context_text}

        ---

        Generate 5 quiz questions from the above context data.'''
        
        response = client.chat.completions.create(
            model="llama3.2:latest",
            temperature=0.2,
            messages=[
                { "role": "system", "content": sysIntr,},
                { "role": "user", "content": usrPrompt,}
            ],
            response_model=MCQArr,
        )
        # print(response.model_dump_json(indent=4))

        final_response = {
            "quiz": response.model_dump(),
            "sources": sources
        }

        # Return combined response
        return jsonify(final_response), 200
        # return jsonify(response.model_dump()), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)