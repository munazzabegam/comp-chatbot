from flask import Flask, render_template, request, jsonify, Response, session
import google.generativeai as genai
import os
from dotenv import load_dotenv
from datetime import timedelta

# Load environment
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "super-secret-key")
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=2)

# Configure Gemini
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash-lite")

# Context about Compass Logistics
COMPANY_CONTEXT = """
You are CompassBot, the official AI assistant for Compass Logistics International (https://www.compasslog.com/).
Only answer questions about Compass Logistics International — its services, locations, and operations.
If a question is unrelated, say: "I'm sorry, I can only answer questions related to Compass Logistics International."

Company Overview:
- Compass Logistics International (CLI) is a global logistics group headquartered in Dubai, UAE.
- Operates 21 locations in 12 countries: Europe, Middle East, Far East, Africa, and North America.
- Services: Air Freight, Sea Freight, Tank & Container Operations, Customs Clearance, Warehousing, and Distribution.
- Values: transparency, trust, commitment, and speed.
"""

def build_prompt(history):
    convo = "\n".join([f"{msg['role'].capitalize()}: {msg['text']}" for msg in history])
    return f"{COMPANY_CONTEXT}\n\n{convo}\nCompassBot:"

@app.route("/")
def home():
    session.permanent = True
    if "history" not in session:
        session["history"] = []
    return render_template("index.html")

@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    """Stream Gemini responses live to the browser."""
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "Please enter a message"}), 400

    # Get conversation history
    session.permanent = True
    history = session.get("history", [])
    history.append({"role": "user", "text": user_input})
    if len(history) > 16:
        history = history[-16:]

    prompt = build_prompt(history)

    def generate():
        try:
            # Stream Gemini content
            for chunk in model.generate_content(prompt, stream=True):
                if chunk and chunk.text:
                    yield f"data: {chunk.text}\n\n"
            # Mark end of stream
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    # Update session after stream
    session["history"] = history
    session.modified = True

    return Response(generate(), mimetype="text/event-stream")

@app.route("/clear", methods=["POST"])
def clear_chat():
    session.pop("history", None)
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
