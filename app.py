from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, fitz
from werkzeug.utils import secure_filename
from docx import Document
import os

# === Flask app ===
app = Flask(__name__)
CORS(app)

# === Lazy-loaded SVM model ===
svm_model = None
def get_svm_model():
    global svm_model
    if svm_model is None:
        svm_model = joblib.load("models/svm/svm_ai_detector.pkl")
    return svm_model

# === Utility to read text ===
def extract_text(file):
    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()

    try:
        if ext == ".txt":
            return file.read().decode("utf-8", errors="ignore")
        elif ext == ".pdf":
            text = ""
            pdf = fitz.open(stream=file.read(), filetype="pdf")
            for page in pdf:
                text += page.get_text()
            return text
        elif ext == ".docx":
            doc = Document(file)
            return "\n".join(p.text for p in doc.paragraphs)
        else:
            raise ValueError("Unsupported file type")
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")

# === Routes ===
@app.route("/detect-ai", methods=["POST"])
def detect_ai():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "File is required"}), 400

    try:
        text = extract_text(file)
        model = get_svm_model()
        prediction = model.predict([text])[0]
        confidence = model.predict_proba([text])[0][1]

        return jsonify({
            "ai_score": float(round(confidence, 4)),
            "human_score": float(round(1 - confidence, 4)),
            "verdict": "Likely AI-generated" if confidence > 0.7 else "Likely Human-written",
            "model": "SVM (TF-IDF)"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/", methods=["GET"])
def home():
    return "Detect AI backend running!"

# === Run using Gunicorn in Render ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
