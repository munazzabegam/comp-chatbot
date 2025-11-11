from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Initialize model
model = genai.GenerativeModel("gemini-2.5-flash")

# Company context to restrict bot answers
COMPANY_CONTEXT = """
You are CompassBot, an AI assistant for Compass Logistics International (https://www.compasslog.com/).
You only answer questions related to this company, its services, locations, and operations.

If the user asks anything unrelated (like technology, programming, or people not in the company),
reply with: "I'm sorry, I can only answer questions related to Compass Logistics International."

Company Overview:
- Compass Logistics International (CLI) is a global logistics group headquartered in Dubai, UAE.
- Operates 21 locations in 12 countries: Europe, Middle East, Far East, Africa, and North America.
- Core services include: Air Freight, Sea Freight, Tank & Container Operations, Customs Clearance, Warehousing, and Distribution.
- Values: transparency, trust, commitment, and speed.
- Website: https://www.compasslog.com/
"""

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    try:
        prompt = f"""{COMPANY_CONTEXT}

User: {user_input}
CompassBot:"""
        response = model.generate_content(prompt)
        reply = response.text.strip()
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
