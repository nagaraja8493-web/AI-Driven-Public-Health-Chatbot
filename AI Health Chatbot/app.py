"""
Flask Web Application for AI-Driven Public Health Chatbot.
Integrates Machine Learning (TF-IDF + Logistic Regression) and
Deep Learning (Bi-LSTM) intent classification pipelines with SQLite database,
comprehensive health topics explorer, admin analytics dashboard, and REST APIs.
"""

import os
import json
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS

from database.database import (
    init_db, register_user, authenticate_user, log_conversation,
    get_user_conversations, get_all_health_topics, get_admin_dashboard_stats
)
from chatbot.predictor import get_predictor
from chatbot.response import EMERGENCY_RESPONSE

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "health-chatbot-secret-key-2026-secure")
CORS(app)

# Ensure database is initialized at start
with app.app_context():
    init_db()

@app.context_processor
def inject_user():
    """Injects current logged-in user into templates."""
    return dict(current_user=session.get("user"))

# -------------------------------------------------------------
# Frontend Page Routes
# -------------------------------------------------------------

@app.route("/")
def index():
    """Landing Home page."""
    topics = get_all_health_topics()[:6]
    return render_template("index.html", featured_topics=topics)

@app.route("/chat")
def chat():
    """Chatbot conversation interface."""
    return render_template("chat.html")

@app.route("/topics")
def topics():
    """Searchable health topics directory."""
    all_topics = get_all_health_topics()
    return render_template("topics.html", topics=all_topics)

@app.route("/about")
def about():
    """Redirect removed about route to home page."""
    return redirect(url_for("index"))

@app.route("/faq")
def faq():
    """Frequently Asked Questions & Emergency Guidelines."""
    return render_template("faq.html")

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    """Separate dedicated login portal for administrators."""
    user = session.get("user")
    if user and user.get("role") == "admin":
        return redirect(url_for("admin"))

    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        auth_user = authenticate_user(email, password)
        if auth_user:
            if auth_user.get("role") == "admin":
                session["user"] = auth_user
                return redirect(url_for("admin"))
            return render_template("admin_login.html", error="Access Denied: Administrator role is required to access this portal.")
        return render_template("admin_login.html", error="Invalid administrative credentials. Please check your email and password.")
    return render_template("admin_login.html")

@app.route("/admin/logout")
def admin_logout():
    """Dedicated admin logout."""
    session.pop("user", None)
    return redirect(url_for("admin_login"))

@app.route("/admin")
def admin():
    """Dedicated Admin analytics and control dashboard (Admin only)."""
    user = session.get("user")
    if not user or user.get("role") != "admin":
        return redirect(url_for("admin_login"))

    stats = get_admin_dashboard_stats()
    
    # Load model comparison metrics if available
    ml_metrics_path = os.path.join(os.path.dirname(__file__), "models", "ml_metrics.json")
    dl_metrics_path = os.path.join(os.path.dirname(__file__), "models", "dl_metrics.json")
    
    ml_metrics = None
    dl_metrics = None
    if os.path.exists(ml_metrics_path):
        with open(ml_metrics_path, "r") as f:
            ml_metrics = json.load(f)
    if os.path.exists(dl_metrics_path):
        with open(dl_metrics_path, "r") as f:
            dl_metrics = json.load(f)

    return render_template("admin.html", stats=stats, ml_metrics=ml_metrics, dl_metrics=dl_metrics)

@app.route("/login", methods=["GET", "POST"])
def login():
    """User login route."""
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        user = authenticate_user(email, password)
        if user:
            session["user"] = user
            if user.get("role") == "admin":
                return redirect(url_for("admin"))
            return redirect(url_for("chat"))
        return render_template("login.html", error="Invalid email or password.")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    """User registration route."""
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        
        if not name or not email or not password:
            return render_template("register.html", error="All fields are required.")
        
        res = register_user(name, email, password)
        if res["success"]:
            session["user"] = {"user_id": res["user_id"], "name": res["name"], "email": res["email"], "role": "user"}
            return redirect(url_for("chat"))
        return render_template("register.html", error=res.get("error", "Registration failed."))
    return render_template("register.html")

@app.route("/logout")
def logout():
    """User logout route."""
    session.pop("user", None)
    return redirect(url_for("index"))

# -------------------------------------------------------------
# REST API Endpoints
# -------------------------------------------------------------

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Primary Chat API.
    Request JSON:
    {
        "message": "What are symptoms of diabetes?",
        "model": "ml" | "dl",
        "threshold": 0.60
    }
    """
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    model_type = data.get("model", "ml").lower()
    threshold = float(data.get("threshold", 0.60))

    if not message:
        return jsonify({"error": "Empty message"}), 400

    predictor = get_predictor()
    result = predictor.get_chat_response(message, model_type=model_type, confidence_threshold=threshold)

    # Determine user_id
    current_user = session.get("user")
    user_id = current_user["user_id"] if current_user else 2 # default guest user

    # Save to SQLite database
    conv_id = log_conversation(
        user_id=user_id,
        message=message,
        intent=result["intent"],
        response=result["response"],
        confidence=result["confidence"],
        model_used=result["model_used"],
        is_emergency=result["is_emergency"]
    )

    result["conversation_id"] = int(conv_id)
    return jsonify(result)

@app.route("/api/topics", methods=["GET"])
def api_topics():
    """Returns all health topics."""
    return jsonify(get_all_health_topics())

@app.route("/api/history", methods=["GET"])
def api_history():
    """Returns conversation history for logged-in user or guest."""
    current_user = session.get("user")
    user_id = current_user["user_id"] if current_user else 2
    history = get_user_conversations(user_id, limit=30)
    return jsonify(history)

@app.route("/api/admin/stats", methods=["GET"])
def api_admin_stats():
    """Returns aggregated admin dashboard statistics."""
    stats = get_admin_dashboard_stats()
    return jsonify(stats)

@app.route("/api/comparison", methods=["GET"])
def api_comparison():
    """Returns model comparison data for ML vs DL."""
    ml_metrics_path = os.path.join(os.path.dirname(__file__), "models", "ml_metrics.json")
    dl_metrics_path = os.path.join(os.path.dirname(__file__), "models", "dl_metrics.json")
    
    ml_data = {}
    dl_data = {}
    if os.path.exists(ml_metrics_path):
        with open(ml_metrics_path, "r") as f:
            ml_data = json.load(f)
    if os.path.exists(dl_metrics_path):
        with open(dl_metrics_path, "r") as f:
            dl_data = json.load(f)

    return jsonify({
        "ml": ml_data,
        "dl": dl_data
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
