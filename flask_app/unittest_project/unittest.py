# import os
# import base64
# from flask import Flask, request, jsonify, render_template
# from werkzeug.utils import secure_filename
# import openai
# from PIL import Image
# import io
# # Initialize Flask app
# app = Flask(__name__)

# # Set OpenAI API Key
# openai.api_key = os.getenv("OPENAI_API_KEY")

# # Configure upload folder and allowed file extensions
# UPLOAD_FOLDER = 'uploads'
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size to 16 MB

# # Ensure the upload folder exists
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# def analyze_image_with_gpt(image_data, prompt):
#     """Analyze an image by first converting it to a description."""
#     try:
#         # Decode image and verify it
#         image = Image.open(io.BytesIO(image_data))
#         image_format = image.format
#         image_size = image.size  # (width, height)

#         # Create a description
#         image_description = (
#             f"The user uploaded an image of type '{image_format}' "
#             f"with dimensions {image_size[0]}x{image_size[1]} pixels. "
#             "Please analyze this image for relevance to the challenge prompt "
#             "and provide feedback on its logic and clarity."
#         )

#         # Messages for OpenAI
#         messages = [
#             {
#                 "role": "system",
#                 "content": (
#                     "You are a senior UX or product design manager reviewing a student's design submission. "
#                     f"The challenge is: '{prompt if prompt else 'No specific challenge provided.'}'. "
#                     "Assess whether the uploaded image description effectively addresses the challenge. "
#                     "Evaluate the design logic, clarity, and its relevance to solving the user's needs."
#                     "if the uploaded images is irrelevant, describe the content of image."
#                     "keep your answer in 3 sentences"
#                 ),
#             },
#             {"role": "user", "content": image_description},
#         ]

#         # Call OpenAI GPT-4
#         response = openai.ChatCompletion.create(
#             model="gpt-4o",
#             messages=messages,
#             temperature=0.6,
#             max_tokens=512,
#         )

#         return response["choices"][0]["message"]["content"]

#     except Exception as e:
#         print(f"Error analyzing image: {e}")
#         return "An error occurred while analyzing the image. Please ensure it is valid."

# @app.route('/')
# def index():
#     """Render the homepage with upload functionality."""
#     return render_template('index.html')


# @app.route("/upload-file", methods=["POST"])
# def upload_file():
#     # global challenge_prompt
#     if 'file' not in request.files:
#         return jsonify(error="No file part"), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify(error="No selected file"), 400

#     try:
#         # Read the file as binary data
#         image_data = file.read()
#                     # Placeholder for a challenge prompt (can be passed dynamically)
#         challenge_prompt = "Design an app for managing tasks efficiently for remote teams."
#         # Analyze the image
#         feedback = analyze_image_with_gpt(image_data, challenge_prompt)

#         return jsonify(feedback=feedback)

#     except Exception as e:
#         print(f"Error during file analysis: {e}")
#         return jsonify(error="An error occurred while analyzing the image."), 500
# if __name__ == '__main__':
#     app.run(debug=True)
    
import os
import base64
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import openai
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Set OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configure upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size to 16 MB

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Helper function to check file type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def analyze_image_with_gpt(image_data_url, user_input, challenge_prompt):
    """Analyze an image using GPT-4o with `image_url`."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior UX or product design manager reviewing a student's design submission. "
                f"The challenge is: '{challenge_prompt if challenge_prompt else 'No specific challenge provided.'}'. "
                "Evaluate whether the uploaded image and/or user input effectively address the challenge. "
                "Assess the design logic, clarity, and relevance to solving the user's needs. "
                "Provide feedback that is motivational and constructive."
                "describe image content"
            ),
        }
    ]

    # Add image_url content if available
    if image_data_url:
        messages.append({
            "role": "user",
            "content": (
                "what's in this image?"
                f"![Uploaded Image]({image_data_url})" ) # Include image using markdown syntax
        })

    # Add user input if provided
    if user_input:
        messages.append({"role": "user", "content": user_input})

    # Send to GPT-4o for analysis
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.6,
        max_tokens=512,
    )

    return response["choices"][0]["message"]["content"]


@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')


@app.route('/upload-file', methods=['POST'])
def upload_file():
    # Placeholder challenge prompt
    challenge_prompt = "Design an app for managing tasks efficiently for remote teams."

    try:
        # Get user input and file
        user_input = request.form.get("user_input", "").strip()
        file = request.files.get("file")

        if not file and not user_input:
            return jsonify(error="Please provide an input or upload an image."), 400

        image_data_url = None
        if file and allowed_file(file.filename):
            # Convert image to base64 URL
            image_data = file.read()
            image_data_url = f"data:image/{file.filename.rsplit('.', 1)[1].lower()};base64," + base64.b64encode(image_data).decode('utf-8')

        # Analyze with GPT-4o
        feedback = analyze_image_with_gpt(image_data_url, user_input, challenge_prompt)
        return jsonify(feedback=feedback)

    except Exception as e:
        print(f"Error during file processing: {e}")
        return jsonify(error="An error occurred while processing your request."), 500


if __name__ == '__main__':
    app.run(debug=True)