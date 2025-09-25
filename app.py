from flask import Flask, request, jsonify, session, send_from_directory
from datetime import timedelta
import uuid
import os
import google.generativeai as genai
import httpx # NEW: Import httpx for making HTTP requests to Ollama

# Import your database and model modules
from database import create_tables, log_conversation, get_conversation_history
from model import NLTKIntentClassifier, TransformerIntentClassifier # Import both

BASE_DIR = os.path.dirname(os.path.abspath(__file__))



app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, 'static'),
    static_url_path='/static',
)
app.secret_key = os.urandom(24) # Replace with a strong, permanent secret key in production
app.permanent_session_lifetime = timedelta(minutes=30) # Session timeout
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


# --- Google Gemini API Configuration ---
# It's best practice to load this from an environment variable, not hardcode
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# Initialize the generative model (choose your desired model)
# Keep this as 'gemini-1.5-flash' or 'text-bison-001' as 'gemini-pro' was giving 404
gemini_model = genai.GenerativeModel('gemini-1.5-flash') # Or 'text-bison-001'
# --- End Gemini API Configuration ---


# --- Ollama API Configuration ---
OLLAMA_API_URL = "http://localhost:11434/api/chat" # Using /api/chat for conversation history
OLLAMA_MODEL_NAME = "phi3" # CHANGE THIS to the model you downloaded (e.g., "mistral", "gemma", "phi3")
# --- End Ollama API Configuration ---


# Initialize the database
create_tables()

# --- Choose your model type (for /chat endpoint) ---
# Set this to 'nltk' or 'transformer' based on which model you trained
MODEL_TYPE = 'transformer'
CONFIDENCE_THRESHOLD = 0.7 # Minimum confidence for a response

if MODEL_TYPE == 'nltk':
    chatbot_model = NLTKIntentClassifier()
    if not chatbot_model.load_model():
        print("NLTK model not found. Please run train_model.py with MODEL_TYPE = 'nltk' first.")
        exit()
elif MODEL_TYPE == 'transformer':
    chatbot_model = TransformerIntentClassifier()
    if not chatbot_model.load_model():
        print("Transformer model not found. Please run train_model.py with MODEL_TYPE = 'transformer' first.")
        exit()
else:
    print("Invalid MODEL_TYPE. Please set to 'nltk' or 'transformer'.")
    exit()

@app.before_request
def make_session_permanent():
    session.permanent = True
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    # print(f"Current session ID: {session['session_id']}")





@app.route('/')
def home():
    # Serve the index.html from the static folder
    return send_from_directory('static', 'index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    session_id = session.get('session_id')

    if not user_message:
        return jsonify({"response": "Please provide a message."}), 400

    if not session_id:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        print(f"New session created for request: {session_id}")

    print(f"Received message from session {session_id}: {user_message}")


    # Predict intent
    intent_tag, confidence = chatbot_model.predict_intent(user_message)

    bot_response = "I'm sorry, I don't understand. Can you please rephrase?"
    if confidence >= CONFIDENCE_THRESHOLD:
        bot_response = chatbot_model.get_response(intent_tag)
    else:
        # Fallback for low confidence
        print(f"Low confidence ({confidence:.2f}) for intent '{intent_tag}'. Using fallback.")

    # Log the conversation
    log_conversation(session_id, user_message, bot_response)

    return jsonify({
        "response": bot_response,
        "intent": intent_tag,
        "confidence": float(confidence)
    })

@app.route('/gemini_chat', methods=['POST'])
def gemini_chat():
    user_message = request.json.get('message')
    session_id = session.get('session_id')

    if not user_message:
        return jsonify({"response": "Please provide a message."}), 400

    if not session_id:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        print(f"New session created for Gemini request: {session_id}")

    print(f"Received Gemini message from session {session_id}: {user_message}")

    try:
        # --- START: Changes for Conversation History ---
        db_history = get_conversation_history(session_id)
        gemini_history = []
        for user_msg, bot_resp_raw, _ in db_history:
            gemini_history.append({'role': 'user', 'parts': [user_msg]})
            bot_resp_clean = bot_resp_raw.replace('[Gemini] ', '') # Clean for Gemini
            if bot_resp_clean:
                gemini_history.append({'role': 'model', 'parts': [bot_resp_clean]})

        chat = gemini_model.start_chat(history=gemini_history)
        response = chat.send_message(user_message)
        bot_response = response.text
        # --- END: Changes for Conversation History ---

        log_conversation(session_id, user_message, f"[Gemini] {bot_response}")

        return jsonify({
            "response": bot_response,
            "model": "Gemini",
            "user_message": user_message
        })
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        log_conversation(session_id, user_message, f"[Gemini Error] {str(e)}")
        # Check if it's a quota error to provide a more specific message
        if "429" in str(e) or "quota" in str(e).lower():
            return jsonify({"response": "Sorry, Gemini's daily quota has been exceeded. Please try again tomorrow or use another chatbot.", "model": "Gemini Quota Exceeded"}), 429
        return jsonify({"response": f"Sorry, I'm having trouble connecting to the AI right now. Error: {str(e)}", "model": "Gemini Error"}), 500


# --- NEW OLLAMA CHAT ENDPOINT ---
@app.route('/ollama_chat', methods=['POST'])
def ollama_chat():
    user_message = request.json.get('message')
    session_id = session.get('session_id')

    if not user_message:
        return jsonify({"response": "Please provide a message."}), 400

    if not session_id:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        print(f"New session created for Ollama request: {session_id}")

    print(f"Received Ollama message from session {session_id}: {user_message}")

    try:
        # Retrieve history from your database for Ollama
        db_history = get_conversation_history(session_id)

        # Format history for Ollama's /api/chat endpoint
        # Ollama expects [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
        ollama_messages = []
        for user_msg_hist, bot_resp_raw_hist, _ in db_history:
            ollama_messages.append({'role': 'user', 'content': user_msg_hist})
            # Clean up [Gemini] or [Ollama] prefixes from bot responses
            bot_resp_clean_hist = bot_resp_raw_hist.replace('[Gemini] ', '').replace('[Ollama] ', '')
            if bot_resp_clean_hist:
                ollama_messages.append({'role': 'assistant', 'content': bot_resp_clean_hist})

        # Add the current user message
        ollama_messages.append({'role': 'user', 'content': user_message})

        payload = {
            "model": OLLAMA_MODEL_NAME, # Use the configured model name
            "messages": ollama_messages, # Send the full conversation history
            "stream": False # Get the full response at once
        }

        # Send request to Ollama local API
        # Using a higher timeout as local LLM inference can take time
        response = httpx.post(OLLAMA_API_URL, json=payload, timeout=300)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        ollama_response = response.json()
        # Ollama /api/chat response structure is {..., "message": {"role": "assistant", "content": "..."}}
        bot_response = ollama_response['message']['content']

        log_conversation(session_id, user_message, f"[Ollama] {bot_response}")

        return jsonify({
            "response": bot_response,
            "model": f"Ollama ({OLLAMA_MODEL_NAME})",
            "user_message": user_message
        })
    except httpx.HTTPStatusError as e:
        error_message = f"Ollama HTTP error: {e.response.status_code} - {e.response.text}"
        print(error_message)
        log_conversation(session_id, user_message, f"[Ollama Error] {error_message}")
        return jsonify({"response": f"Sorry, I'm having trouble with the local AI. Error: {error_message}", "model": "Ollama Error"}), 500
    except httpx.RequestError as e:
        error_message = f"Ollama Network error: {e}"
        print(error_message)
        log_conversation(session_id, user_message, f"[Ollama Error] {error_message}")
        return jsonify({"response": f"Sorry, I'm having trouble reaching the local AI (Is Ollama running and model loaded?). Error: {error_message}", "model": "Ollama Error"}), 500
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        print(error_message)
        log_conversation(session_id, user_message, f"[Ollama Error] {error_message}")
        return jsonify({"response": f"Sorry, something went wrong with the AI. Error: {error_message}", "model": "Ollama Error"}), 500


@app.route('/history', methods=['GET'])
def get_history():
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({"message": "No active session found."}), 400

    history = get_conversation_history(session_id)
    formatted_history = []
    for user_msg, bot_resp, timestamp in history:
        formatted_history.append({
            "user_message": user_msg,
            "bot_response": bot_resp,
            "timestamp": timestamp
        })
    return jsonify({"session_id": session_id, "history": formatted_history})

@app.route('/reset', methods=['POST'])
def reset_session():
    session_id = session.pop('session_id', None)
    if session_id:
        print(f"Session {session_id} reset.")
        return jsonify({"message": f"Session {session_id} reset successfully."})
    else:
        return jsonify({"message": "No active session to reset."})
    

if __name__ == '__main__':
    # For production, use Gunicorn or Uvicorn. For development:
    app.run(debug=True, host='0.0.0.0', port=5000)
