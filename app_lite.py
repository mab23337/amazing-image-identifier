from flask import Flask, jsonify
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)

@app.errorhandler(Exception)
def handle_exception(e):
    logging.exception("Unhandled backend error:")
    return jsonify({"error": "Internal server error"}), 500

@app.route("/")
def home():
    return "Backend running (lite)!"

@app.route("/analyze", methods=["POST"])
def analyze():
    return jsonify({"message": "AI deps not installed on Pi yet; backend scaffold OK."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
