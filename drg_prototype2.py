import openai
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
from gtts import gTTS
import os
import base64
import uuid
import subprocess
import io
from PIL import Image
import pytesseract  # For extracting text from images
# Initialize OpenAI API
openai.api_key = st.secrets["OPENAI_API_KEY"]
# Predefined leading questions
PREDEFINED_QUESTIONS = [
    "Help me understand the context",
    "Who is the user?",
    "What are the users' pain point",
    "Are there any restrictions or constraints to consider?"
]

QUESTION_KEYS = {
    "context_question": "Help me understand the context",
    "user_question": "Who is the user?",
    "problem_question": "What are the users' pain point",
    "constraints_question": "Are there any restrictions or constraints to consider?"
}

IDEATE_KEYS={
    "I am ready to ideate": "Can you help me summarize the goal, user groups, and painponts?",
    "I have more questions": "I have more questions"
}

# Function to handle TTS and generate an audio file
def generate_audio(text, slow_down=False):
    """Generate audio using gTTS with enhanced naturalness."""
    formatted_text = format_text_for_natural_pauses(text)
    tts = gTTS(formatted_text)
    audio_filename = f"audio_{uuid.uuid4().hex}.mp3"
    tts.save(audio_filename)

    # Optionally slow down the audio
    if slow_down:
        slowed_audio_filename = f"slowed_{uuid.uuid4().hex}.mp3"
        slow_down_audio(audio_filename, slowed_audio_filename)
        os.remove(audio_filename)  # Remove original
        audio_filename = slowed_audio_filename

    # Convert audio to base64
    with open(audio_filename, "rb") as audio_file:
        audio_bytes = audio_file.read()
    os.remove(audio_filename)  # Clean up audio file
    return base64.b64encode(audio_bytes).decode()
# Function to embed an audio player in Streamlit
# Function to embed an audio player in Streamlit with a stop mechanism
def play_audio(audio_base64):
    """Embed an audio player in Streamlit to play the generated audio, stopping the previous one."""
    # Inject JavaScript to stop any existing audio playback
    st.markdown(
        """
        <script>
        const oldAudio = document.getElementById("ai_audio");
        if (oldAudio) {
            oldAudio.pause();
            oldAudio.currentTime = 0; // Reset the playback
        }
        </script>
        """,
        unsafe_allow_html=True,
    )
    # Add the new audio player
    st.markdown(
        f"""
        <audio id="ai_audio" autoplay>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
        """,
        unsafe_allow_html=True,
    )
# Function to enhance text for gTTS to sound more natural
def format_text_for_natural_pauses(text):
    """Enhance text for natural pauses in gTTS."""
    text = text.replace(';', '; ')
    text = text.replace('.', '. ')
    # text = text.replace('...', '. . .')
    return text
# Function to display a typing effect for the AI's blue bubble responses
def display_typing_effect(response_text):
    """Display text with a typing effect in the AI's response bubble."""
    response_placeholder = st.empty()
    for i in range(len(response_text) + 1):
        response_placeholder.markdown(
            f'<div class="message-bubble ai-message">{response_text[:i]}</div>',
            unsafe_allow_html=True,
        )
        time.sleep(0.05)  # Adjust typing speed

# Function to slow down audio playback using FFmpeg
def slow_down_audio(input_file, output_file, speed=0.85):
    """Slow down audio playback using FFmpeg."""
    subprocess.run(["ffmpeg", "-i", input_file, "-filter:a", f"atempo={speed}", "-vn", output_file])
# Function to get embeddings for a given text
def get_embeddings(text):
    """Fetch embeddings for a given text using OpenAI's embedding model with input validation."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text for embeddings must be a non-empty string.")
    response = openai.Embedding.create(
        input=[text],  # Ensure input is passed as a list
        model="text-embedding-ada-002"
    )
    return np.array(response['data'][0]['embedding'])
# Function to check for similar questions
def is_similar_to_previous_questions(new_question, previous_questions, threshold=0.85):
    """
    Check if the new question is semantically similar to any previous questions.
    Uses cosine similarity with OpenAI embeddings.
    """
    if not previous_questions:
        return False
    new_embedding = get_embeddings(new_question)
    previous_embeddings = [get_embeddings(q) for q in previous_questions]  # Directly use strings
    similarities = cosine_similarity([new_embedding], previous_embeddings)[0]
    return max(similarities) >= threshold

# Check user input against predefined questions and mark as answered if similar
def check_input_similarity_and_remove_buttons(user_input):
    """
    Checks if user input matches any predefined questions semantically,
    removes corresponding buttons, and updates session state.
    """
    # Extract previous questions from the conversation
    previous_questions = [msg["content"] for msg in st.session_state["conversation"] if msg["role"] == "user"]

    for question_key, question_text in PREDEFINED_QUESTIONS:
        if question_key not in st.session_state["answered_questions"] and is_similar_to_previous_questions(user_input, previous_questions):
            # Mark the question as answered
            st.session_state["answered_questions"].add(question_key)
            # Append user input and AI response to conversation history
            st.session_state["conversation"].append({"role": "user", "content": user_input})
            st.session_state["conversation"].append({
                "role": "assistant",
                "content": f"You've already asked about '{question_text}'. Let's build on that."
            })
            return True
        
    return False

# Functions for GPT-4 interactions
def get_ai_response(user_input, prompt=None):
    """Get AI response based on user input and optional prompt."""
    messages = [
        {"role": "system", "content": "You are acting like a senior UX or product design manager that is going to or already give your interviewee whiteboard design challenges." 
         f"The challenge is: '{st.session_state['challenge_prompt']}'" 
         "Your answer to whatever users question should not convey specific solution. but it should give some directions help them understand the prompt and have enough information" 
         "You should remeber what they have asked because later when you are giving feedback on how well they did for this challenge, you can reflect how they are process and taclke through what questions they have asked"
         "keep your answer less than 5 sentences"
         "dont throw jargons"
         }
    ]
    if prompt:
        messages.append({"role": "system", "content": f"The challenge is: '{prompt}'"})
    messages.append({"role": "user", "content": user_input})
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7,
        max_tokens=4096
    )
    
    return response["choices"][0]["message"]["content"]
# Function to generate repetitive question response
def generate_repetitive_question_response(user_input):
    """Generate AI response when a user asks a similar question."""
    messages = [
        {"role": "system", "content": "You are an senior designer from big tech helping your interviewee navigate a whiteboard design challenge. "
         "The interviewee has asked a similar question before. Remind them stop asking same questions in friendly way. Keep your message in 1-2 sentences and use oral words"},
        {"role": "user", "content": f"The user has already asked this question: '{user_input}'. How can I stop them asking same questions and move forward"}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7,
        max_tokens=256
    )
    return response["choices"][0]["message"]["content"]

def generate_challenge():
    """Generate a whiteboard challenge prompt."""
    with st.spinner("Generating your challenge..."):
        time.sleep(2)  # Simulate processing delay
        prompt_generation_instruction = (
            "You are acting like a senior UX or product design manager giving a whiteboard challenge to your interviewees. "
            "The challenge can be an app, a service, or any platform"
            "The challenge is about users in real life"
            "The challenge should be something reltively common seen and cutting-edge"
            "The challenge prompt should be chosen from varied topics but avoid technical jargons"
            "The challenge prompt should not limited to certain platform"
            "Keep your prompt in 3-5sentences and one paragraph"
        )
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": prompt_generation_instruction}],
            temperature=0.8,
            max_tokens=256
        )
        return response["choices"][0]["message"]["content"]

# Function to handle individual question responses
def handle_question_response(question_key, question_text):
    """Handle response for predefined question buttons."""
    with st.spinner(f"Processing '{question_text}'..."):
        time.sleep(1)  # Simulate processing delay
        ai_response = get_ai_response(question_text, st.session_state["challenge_prompt"])
        
        # Mark question as answered and add to conversation
        st.session_state["answered_questions"].add(question_key)
        st.session_state["conversation"].append({"role": "user", "content": question_text})
        st.session_state["conversation"].append({"role": "assistant", "content": ai_response})
        
        # Generate TTS and play audio
        st.session_state["audio_base64"] = generate_audio(ai_response)    
        play_audio(st.session_state["audio_base64"])

    check_all_questions_answered()

def handle_generate_button_click():
    st.session_state["button_clicked"] = True
    st.session_state["challenge_prompt"] = generate_challenge()
    audio_base64 = generate_audio(st.session_state["challenge_prompt"]) 
    st.session_state["audio_base64"] = audio_base64  # Store audio in session state
    play_audio(st.session_state["audio_base64"])

# Function to record asked questions
def record_question(question):
    st.session_state["answered_questions"].add(question)
    st.session_state["conversation"].append({"role": "user", "content": question})

# Function to check if all leading questions have been answered
# Function to check if all leading questions have been answered
def check_all_questions_answered():
    """Check if all predefined questions have been answered."""
    # Transition to ideation phase if all questions are answered
    if st.session_state["answered_questions"] == set(QUESTION_KEYS.keys()):
        st.session_state["ready_for_ideation"] = True  # Trigger ideation phase
        # Add the conversation responses
        ideation_start_messages = [
            "It looks like we've covered all the key questions. Great work! Now, let's move on to ideating.",
            "You can now summarize your findings or start ideating key features and flows for the design challenge."
        ]
        for message in ideation_start_messages:
            st.session_state["conversation"].append({
                "role": "assistant",
                "content": message
            })
        
        # Generate and play the audio for the responses
        combined_message = " ".join(ideation_start_messages)
        audio_base64 = generate_audio(combined_message, slow_down=True)  # Optionally slow down for clarity
        st.session_state["audio_base64"] = audio_base64
        play_audio(st.session_state["audio_base64"])

def handle_user_input():
    """Process user input and add to conversation."""
    user_message = st.session_state["user_input"]  # Get the input from session state
    if user_message:  # Ensure there's input
        # Check for similarity with predefined questions
        for idx, question in enumerate(PREDEFINED_QUESTIONS):
            question_key = list(QUESTION_KEYS.keys())[idx]  # Map predefined question to its key
            if question_key not in st.session_state["answered_questions"]:
                # Check for semantic similarity
                if is_similar_to_previous_questions(user_message, [question]):
                    # Mark question as answered and remove corresponding button
                    st.session_state["answered_questions"].add(question_key)
                    st.session_state["conversation"].append({"role": "user", "content": user_message})
                    ai_response = get_ai_response(question, st.session_state["challenge_prompt"])                                      
                    st.session_state["conversation"].append({"role": "assistant", "content": ai_response})
                    # Generate TTS and play audio
                    st.session_state["audio_base64"] = generate_audio(ai_response)
                    play_audio(st.session_state["audio_base64"])
                    display_typing_effect(ai_response)
                    check_all_questions_answered()
                    st.session_state["user_input"] = ""  # Clear input
                    return  # Exit to prevent further processing

        # Check if the user input is repetitive in the conversation
        if is_similar_to_previous_questions(
            user_message,
            [msg["content"] for msg in st.session_state["conversation"] if msg["role"] == "user"]
        ):
            # Generate repetitive question response
            st.session_state["conversation"].append({"role": "user", "content": user_message})
            ai_response = generate_repetitive_question_response(user_message)         
            st.session_state["conversation"].append({"role": "assistant", "content": ai_response})
            # Generate TTS and play audio
            st.session_state["audio_base64"] = generate_audio(ai_response)
            play_audio(st.session_state["audio_base64"])
        else:
            # Handle as a new input if not repetitive
            st.session_state["conversation"].append({"role": "user", "content": user_message})
            ai_response = get_ai_response(user_message)
            
            st.session_state["conversation"].append({"role": "assistant", "content": ai_response})
            # Generate TTS and play audio
            st.session_state["audio_base64"] = generate_audio(ai_response)
            play_audio(st.session_state["audio_base64"])
        # Clear the input field
        st.session_state["user_input"] = ""  # Reset the input field in session state

# Function to analyze uploaded image and provide feedback
def analyze_sketch(image, prompt):
    """
    Analyze the uploaded sketch and provide feedback.
    - If the image is not related to the prompt, provide a warning.
    - If the image is related, give detailed feedback.
    """
    try:
        # Extract text from the image (for basic understanding)
        extracted_text = pytesseract.image_to_string(image)

        # Prepare messages for GPT-4
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior UX or product design manager. A user has uploaded a sketch of their user flow, "
                    "and you need to determine if it is related to the design challenge prompt. If it is not, respond "
                    "with 'Please upload the user flow, low-fi picture about the prompt, so that I can give you feedback.' "
                    "If it is related, provide thoughtful feedback based on the sketch content and the design prompt."
                )
            },
            {"role": "user", "content": f"The design prompt is: {prompt}"},
            {"role": "user", "content": f"The text extracted from the sketch is: {extracted_text}"}
        ]

        # Use GPT-4 to analyze relevance and provide feedback
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=200
        )
        return response["choices"][0]["message"]["content"]

    except Exception as e:
        return f"An error occurred while processing the image: {str(e)}"

# Function to handle file upload and analysis
def handle_sketch_upload():
    st.markdown("### Upload Your Sketch")
    uploaded_file = render_sketch_upload_ui()  # Use the updated UI function

    if uploaded_file is not None:
        # Load the uploaded image
        image = Image.open(uploaded_file)

        # Display the uploaded image
        st.image(image, caption="Uploaded Sketch", use_column_width=True)

        # Analyze the sketch and provide feedback
        feedback = analyze_sketch(image, st.session_state.get("challenge_prompt", ""))
        st.markdown(f"**AI Feedback:** {feedback}")
        
def render_sketch_upload_ui():
    """
    Render the drag-and-drop file upload UI for the sketch.
    """
    st.markdown(
        """
        <div class="file-upload-container">
            <form>
                <label for="file-upload">Drag and drop your sketch here, or click to browse.</label>
                <input type="file" id="file-upload" accept="image/png, image/jpeg">
            </form>
            <div class="add-icon">+</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return st.file_uploader("Upload your sketch", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
# Initialize session state
if "conversation" not in st.session_state:
    st.session_state["conversation"] = []
if "challenge_prompt" not in st.session_state:
    st.session_state["challenge_prompt"] = None
if "button_clicked" not in st.session_state:
    st.session_state["button_clicked"] = False  # Track if the generate button was clicked
if "answered_questions" not in st.session_state:
    st.session_state["answered_questions"] = set()  # Track answered leading questions
if "ready_for_ideation" not in st.session_state:
    st.session_state["ready_for_ideation"] = False
if "audio_base64" not in st.session_state:
    st.session_state["audio_base64"] = None
# CSS for chat layout
st.markdown(
    """
    <style>
        .chat-container {
            max-width: 80%;
            margin: 0 auto;
        }
        .message-bubble {
            padding: 10px 15px;
            margin: 10px 10px;
            border-radius: 15px;
            max-width: 70%;
        }
        .user-message {
            background-color: #d1f7c4;
            align-self: flex-end;
            color: #111111;
        }
        .ai-message {
            background-color: #ADD8E6;
            align-self: flex-start;
            color: #111111;
        }
        .prompt-message {
            background-color: transparent;
            border-left: 4px solid #ADD8E6;
            padding: 10px 15px;
            margin: 10px 10px;
            font-weight: bold;
            color: #333333;
        }
        .message-container {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
        }
        .leading-questions-title {
            font-size: 0.9em;
            font-style: italic;
            color: darkgrey;
            margin-bottom: 10px;
        }
        
    .file-upload-container {
        position: relative;
        width: 100%;
        max-width: 600px;
        margin: 20px auto;
        border: 2px dashed #ccc;
        border-radius: 8px;
        text-align: center;
        padding: 30px;
        background-color: #f9f9f9;
        color: #333;
        transition: background-color 0.3s ease;
    }

    .file-upload-container:hover {
        background-color: #f1f1f1;
    }

    .file-upload-container input[type="file"] {
        display: none;
    }

    .file-upload-container label {
        cursor: pointer;
        font-size: 1rem;
        color: #0073e6;
    }

    .file-upload-container img {
        max-width: 80%;
        height: auto;
        display: block;
        margin: 10px auto;
    }

    .add-icon {
        position: absolute;
        bottom: 10px;
        left: 10px;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        background-color: #0073e6;
        color: #fff;
        font-size: 20px;
        line-height: 24px;
        text-align: center;
        font-weight: bold;
    }       
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("Whiteboard Challenge Assistant")
# st.markdown('div class="greeting-message')
# Display the challenge prompt at the top
# Play audio for the last AI response
if st.session_state["audio_base64"]:
    play_audio(st.session_state["audio_base64"])
if st.session_state["challenge_prompt"]:
    st.markdown('<div class="prompt-message">', unsafe_allow_html=True)
    st.markdown(st.session_state["challenge_prompt"], unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Render the "Generate Whiteboard Challenge" button
if not st.session_state["button_clicked"]:
    st.button("Generate Whiteboard Challenge", on_click=handle_generate_button_click)

# Render chat history
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for msg in st.session_state["conversation"]:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="message-container"><div class="message-bubble user-message">{msg["content"]}</div></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="message-container"><div class="message-bubble ai-message">{msg["content"]}</div></div>',
            unsafe_allow_html=True
        )
st.markdown('</div>', unsafe_allow_html=True)

# Define the 4 leading questions with separate handlers
if not st.session_state["ready_for_ideation"] and st.session_state["challenge_prompt"]:
    st.markdown('<div class="leading-questions-title">You may want to ask...</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        # Handle "Help me understand the context"
        if "context_question" not in st.session_state["answered_questions"]:
            st.button(
                "Help me understand the context", 
                on_click=lambda: handle_question_response("context_question", "Help me understand the context")
            )

        # Handle "Who is the user?"
        if "user_question" not in st.session_state["answered_questions"]:
            st.button(
                "Who is the user?", 
                on_click=lambda: handle_question_response("user_question", "Who is the user?")
            )

    with col2:
        # Handle "What problem are you trying to solve?"
        if "problem_question" not in st.session_state["answered_questions"]:
            st.button(
                "What are the users' pain point", 
                on_click=lambda: handle_question_response("problem_question", "What are the users' pain point")
            )
        # Handle "Are there any restrictions or constraints to consider?"
        if "constraints_question" not in st.session_state["answered_questions"]:
            st.button(
                "Are there any restrictions or constraints to consider?", 
                on_click=lambda: handle_question_response("constraints_question", "Are there any restrictions or constraints to consider?")
            )

def handle_start_ideating():
    """Handler for 'I am ready to start ideating' button."""
    # Update conversation with user's decision
    st.session_state["conversation"].append({
        "role": "user",
        "content": "I am ready to start ideating."
    })
    # Add AI response with brainstorming template
    st.session_state["conversation"].append({
        "role": "assistant",
        "content": (
            "Great! Let’s brainstorm. Here’s a simple template to guide you:\n\n"
            "**1. Goals**: What do you aim to achieve?\n"
            "**2. Users**: Who are the target users? What are their characteristics?\n"
            "**3. Pain Points**: What problems are you solving for the users?\n\n"
            "Take your time to fill these out based on what we’ve discussed."
        )
    })
    # Disable the ideation buttons
    st.session_state["ready_for_ideation"] = False

def handle_more_questions():
    """Handler for 'I have more questions' button."""
    # Update conversation with user's decision
    st.session_state["conversation"].append({
        "role": "user",
        "content": "I have more questions."
    })
    # Add AI response to continue with the prompt
    st.session_state["conversation"].append({
        "role": "assistant",
        "content": (
            "Great, we can still work on understanding the prompt further. "
            "Feel free to ask your next question!"
        )
    })
    # Disable the ideation buttons
    st.session_state["ready_for_ideation"] = False

# Transition to ideation phase
if st.session_state["ready_for_ideation"]:
    st.markdown("### Ready to Ideate?")
    col1, col2 = st.columns(2)
    
    with col1:
        # Button for starting ideation
        if st.button("I am ready to start ideating"):
            handle_start_ideating()

    with col2:
        # Button for asking more questions
        if st.button("I have more questions"):
            handle_more_questions()
# Play audio for the last AI response
if st.session_state["audio_base64"]:
    play_audio(st.session_state["audio_base64"])

user_input=st.text_input(
    "Type your message here...", 
    key="user_input",  # Bind input field to session state
    on_change=handle_user_input,  # Trigger handler when input changes (user presses enter)
    placeholder="Type your message and press Enter..."
)

if user_input:
    st.session_state["conversation"].append({"role": "user", "content":  user_input})



