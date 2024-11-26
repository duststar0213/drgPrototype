from flask import Flask, render_template, request, jsonify
import openai
import uuid
import os
import numpy as np
from gtts import gTTS
from sklearn.metrics.pairwise import cosine_similarity
import time
import random
import logging

# Initialize Flask app
app = Flask(__name__)

# Set your OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Predefined questions and keys
PREDEFINED_QUESTIONS = [
    "Help me understand the context",
    "Who is the user?",
    "What are the users pain points?",
    "Are there any restrictions or constraints to consider?"
]

QUESTION_KEYS = {
    "context_question": "Help me understand the context",
    "user_question": "Who is the user?",
    "problem_question": "What are the users pain points?",
    "constraints_question": "Are there any restrictions or constraints to consider?"
}
# Reference phrases for readiness
READINESS_PHRASES = [
    "I am ready",
    "Let's move forward",
    "I am ready to ideate",
    "Start ideation",
    "I want to begin designing",
    "Let's start the next phase",
    "I have enough information",
    "This is enough for me"
]
# State tracking
conversation = []
answered_questions = set()
challenge_prompt = None
ready_for_questions = False

# Initialize leading_questions globally
# Initialize leading_questions globally
leading_questions = [
    {"question": q, "answered": False} for q in PREDEFINED_QUESTIONS
]


# Fun

# Generate audio from text
def generate_audio(text):
    tts = gTTS(text)
    audio_filename = f"static/audio_{uuid.uuid4().hex}.mp3"
    tts.save(audio_filename)
    return audio_filename

# Function to fetch AI response
def get_ai_response(user_input, prompt=None):
    """Get AI response based on user input and optional prompt."""
    messages = [
        {"role": "system", "content": (
            "You are now acting as a senior UX or product design manager giving whiteboard design challenges. "
            "Your user is a design student who might be new to whiteboard challenges. "
            f"The challenge is: '{prompt if prompt else 'No specific challenge provided.'}'. "
            "Your main goal is to help the user clarify their understanding of the challenge and approach it methodically. "
            "Always encourage them to use the predefined questions to explore the challenge further. "
            f"The predefined questions currently are: {leading_questions}. "
            "If the user seems stuck or provides general input, redirect them to interact with the predefined buttons for guidance. "
            "Keep your response short (less than 3 sentences) but motivational and supportive while encouraging exploration."
        )}
    ]
    
    messages.append({"role": "user", "content": user_input})

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7,
        max_tokens=4096
    )

    return response["choices"][0]["message"]["content"]

# Function to generate a challenge
def generate_challenge():
    """Generate a whiteboard challenge prompt."""
    prompt_generation_instruction = (
        "You are a senior UX or product design manager giving a whiteboard challenge. "
        "The challenge should be an app, service, or platform addressing real user needs. "
        "It should be feasible within 30 minutes of design work. Avoid technical jargon. "
        "The challenge should target on no more 3 users groups."
        "Keep the prompt concise, with 3-5 sentences."
    )
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt_generation_instruction}],
        temperature=0.8,
        max_tokens=256
    )
    return response["choices"][0]["message"]["content"]

# Fetch embeddings for semantic similarity
def get_embeddings(text):
    response = openai.Embedding.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return np.array(response['data'][0]['embedding'])

# Check if a question is semantically similar to a previous one
def is_similar_to_previous_questions(new_question, previous_questions, threshold=0.85):
    if not previous_questions:
        return False
    new_embedding = get_embeddings(new_question)
    previous_embeddings = [get_embeddings(q) for q in previous_questions]
    similarities = cosine_similarity([new_embedding], previous_embeddings)[0]
    return max(similarities) >= threshold
# Function to generate a new leading question
def generate_new_question(excluded_questions, answered_questions):
    prompt = (
        "You are a senior UX or product design manager. Based on a design challenge, generate a leading question "
        "that will help clarify the prompt. Avoid questions about previously answered topics or similar questions. "
        f"Exclude these questions: {excluded_questions}. Previously answered: {answered_questions}. "
        "Focus on clarifying the goal, users, or constraints. Limit the question to one sentence. less than 15 words"
    )
    print(f"Prompt to AI: {prompt}")  # Debug log

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.7,
        max_tokens=100,
    )
    new_question = response["choices"][0]["message"]["content"]
    print(f"New question from AI: {new_question}")  # Debug log
    return new_question
def check_adequate_information(conversation):
    """
    Check if the conversation semantically covers user, pain points, and goal topics.
    """
    covered = {"user": False, "pain_points": False, "goal": False}

    # Reference phrases for semantic similarity
    reference_phrases = {
        "user": ["Who are the users?", "Define the audience", "Who will use this?", "Target customer"],
        "pain_points": ["What problems are being solved?", "Pain points", "Challenges", "User difficulties"],
        "goal": ["What is the goal?", "Purpose of the design", "Objective", "Design aim"]
    }

    for msg in conversation:
        if msg["role"] == "user" or msg["role"] == "assistant":
            content = msg["content"].lower()
            for key, phrases in reference_phrases.items():
                if not covered[key]:
                    # Compare the current content with reference phrases
                    if is_similar_to_any(content, phrases):
                        covered[key] = True

    return all(covered.values())  # Return True if all topics are covered


def is_similar_to_any(text, reference_phrases, threshold=0.85):
    """
    Check if a given text is semantically similar to any reference phrases.
    Uses cosine similarity for comparison.
    """
    text_embedding = get_embeddings(text)
    reference_embeddings = [get_embeddings(phrase) for phrase in reference_phrases]
    similarities = cosine_similarity([text_embedding], reference_embeddings)[0]
    return any(similarity >= threshold for similarity in similarities)
def has_meaningful_interaction(conversation):
    """
    Check if the user has engaged in meaningful interaction with the challenge.
    Meaningful interaction is defined as the user asking a question relevant to the challenge.
    """
    for msg in conversation:
        if msg["role"] == "user":
            # Check if the user's message is a question or resembles a predefined question
            if "?" in msg["content"] or is_similar_to_any(msg["content"], PREDEFINED_QUESTIONS):
                return True  # Found meaningful interaction
    return False
# Routes
@app.route("/")
def index():
    global challenge_prompt, leading_questions
    # Filter unanswered questions to send to the frontend
    predefined_questions = [q["question"] for q in leading_questions]
    unanswered_questions = [q["question"] for q in leading_questions if not q["answered"]]if challenge_prompt else []
    return render_template(
        "index.html",
        conversation=conversation,
        questions=unanswered_questions,
        challenge_prompt=challenge_prompt,
        predefined_questions=predefined_questions
    )

@app.route("/generate-challenge", methods=["POST"])
def generate_challenge_route():
    global challenge_prompt, ready_for_questions, answered_questions

    challenge_prompt = generate_challenge()
    ready_for_questions = True
    answered_questions.clear()  # Reset answered questions
    conversation.append({"role": "assistant", "content": f"Challenge: {challenge_prompt}"})
    return jsonify({"challenge_prompt": challenge_prompt})

@app.route("/process", methods=["POST"])
def process_input():
    global answered_questions, challenge_prompt, leading_questions
    user_input = request.form.get("user_input")
    response = ""
    
       # Check if the user input is similar to readiness phrases
    if is_similar_to_any(user_input, READINESS_PHRASES):
        # Generate AI response for ideation phase
         # Check if the user has engaged with leading questions
        if not has_meaningful_interaction(conversation):
            # Generate a dynamic response using the AI
            prompt = (
        "The user has expressed readiness to proceed without engaging with any of the clarifying questions. "
        "Generate a polite and encouraging response that motivates the user to use the predefined clarifying questions "
        "displayed on the interface or to type their own clarifying questions in the input field. "
        "Emphasize the importance of asking clarifying questions in real-life design challenges, as it helps "
        "narrow down the scope and demonstrates a thoughtful, structured approach. "
        "Keep the tone friendly, motivational, and focused on the value of effective clarification."
            )
            response = get_ai_response(prompt)
            return jsonify({
                "response": response,
                "show_leading_questions": True,
                "hide_ideation_buttons": True,
                "new_questions": [q["question"] for q in leading_questions if not q["answered"]]
                
            })
        else:

            response = (
            "Great! Let's summarize the key points you've gathered so far:<br><br>"
            "<ul>"
            "  <li><strong>Users:</strong>&nbsp;[Who are the users?]</li>"
            "  <li><strong>Goals:</strong>&nbsp;[What is the primary goal?]</li>"
            "  <li><strong>Pain Points:</strong>&nbsp;[What problems are being solved?]</li>"
            "</ul><br>"
            "Take a moment to fill in these details based on your understanding, "
            "and then we can start building the user flow!"
            )
            # Hide leading questions and enable ideation
            return jsonify({"response": response, "ready_to_ideate": True})

    # Check if the user input is similar to any of the current leading questions
    current_questions = [q["question"] for q in leading_questions if not q["answered"]]
    if is_similar_to_any(user_input, current_questions):
        # Find the similar question
        for i, question in enumerate(leading_questions):
            if not question["answered"] and is_similar_to_previous_questions(user_input, [question["question"]]):
                question["answered"] = True  # Mark the similar question as answered
                answered_questions.add(question["question"])  # Track it as answered

                # Generate a new question to replace it
                excluded_questions = [q["question"] for q in leading_questions]
                new_question = generate_new_question(excluded_questions, list(answered_questions))
                leading_questions[i] = {"question": new_question, "answered": False}

                print(f"Replaced similar question '{question['question']}' with '{new_question}'")
                break

    # Handle predefined questions
    if user_input in PREDEFINED_QUESTIONS:
        question_key = list(QUESTION_KEYS.keys())[PREDEFINED_QUESTIONS.index(user_input)]
        if question_key not in answered_questions:
            answered_questions.add(question_key)
            response = get_ai_response(user_input, challenge_prompt)
        else:
            response = "You've already asked this question. Let's move forward."

    # Handle general user input
    else:
        response = get_ai_response(user_input, challenge_prompt)

    # Add the user input and AI response to the conversation
    conversation.append({"role": "user", "content": user_input})
    conversation.append({"role": "assistant", "content": response})

    # Check if adequate information has been provided
    if check_adequate_information(conversation):
        # AI suggests moving forward
        response += " I think you have obtained enough information. What about moving forward?"
        return jsonify({
            "response": response,
            "ready_to_ideate": True,
            "new_questions": [q["question"] for q in leading_questions if not q["answered"]]
        })

    return jsonify({
        "response": response,
        "ready_to_ideate": False,
        "new_questions": [q["question"] for q in leading_questions if not q["answered"]]
        
    })
@app.route("/process-question", methods=["POST"])
def process_question():
    global leading_questions, answered_questions

    # Get the clicked question from the frontend
    clicked_question = request.form.get("clicked_question")
    print(f"Clicked question: {clicked_question}")

    # Mark the clicked question as answered
    for question in leading_questions:
        if question["question"] == clicked_question:
            question["answered"] = True
            answered_questions.add(clicked_question)
            print(f"Marked as answered: {clicked_question}")

    # Check if all questions are answered
    if all(q["answered"] for q in leading_questions):
        print("All questions answered. Ready for ideation.")
        return jsonify({"ready_to_ideate": True})

    # Generate a new question that isn't similar to existing ones
    excluded_questions = [q["question"] for q in leading_questions]
    print(f"Excluded questions: {excluded_questions}")
    new_question = generate_new_question(excluded_questions, list(answered_questions))
    print(f"Generated new question: {new_question}")

    # Replace the clicked question with the new one
    for i, question in enumerate(leading_questions):
        if question["question"] == clicked_question:
            leading_questions[i] = {"question": new_question, "answered": False}
            print(f"Replaced question: {clicked_question} with {new_question}")
            break

    # Return the updated list of unanswered questions
    updated_questions = [q["question"] for q in leading_questions if not q["answered"]]
    print(f"Updated leading questions: {updated_questions}")
    return jsonify({"new_questions": updated_questions})
@app.route("/process-ideation", methods=["POST"])
def process_ideation():
    action = request.form.get("action")  # Get the action from the frontend
    response = ""

    if action == "ready_to_ideate":
        # Generate the ideation summary prompt
        response = (
        "Great! Let's summarize the key points you've gathered so far:<br><br>"
        "<ul>"
        "  <li><strong>Users:</strong> Who are the users?</li>"
        "  <li><strong>Goals:</strong> What is the primary goal?</li>"
        "  <li><strong>Pain Points:</strong> What problems are being solved?</li></ul>"
     
        "<br>Take a moment to fill in these details based on your understanding, "
        "and then we can start building the user flow!"
        )
    elif action == "more_questions":
        # Generate the response encouraging the user to save time for ideation
        response = (
            "Sure, feel free to ask more questions to clarify your understanding. "
            "Remember, in real-life scenarios, it's essential to allocate time "
            "for ideation and creating user flows. Let me know how I can assist!"
        )

    # Return the new response and updated leading questions
    updated_questions = [q["question"] for q in leading_questions if not q["answered"]]
    return jsonify({"response": response, "updated_questions": updated_questions})
@app.route("/audio", methods=["POST"])
def generate_audio_response():
    response_text = request.form.get("response_text")
    audio_file = generate_audio(response_text)
    return jsonify({"audio_file": audio_file})

if __name__ == "__main__":
    app.run(debug=True)