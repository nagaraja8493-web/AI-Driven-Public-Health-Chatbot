"""
Database module for AI Public Health Chatbot.
Handles SQLite connection, schema creation, data seeding, conversation logging,
user authentication, and admin metrics queries.
"""

import sqlite3
import os
import hashlib
from datetime import datetime
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database", "health_chatbot.db")

def hash_password(password: str) -> str:
    """Hashes a password with SHA-256."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def get_db_connection():
    """Returns a SQLite connection with row factory enabled."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database schema and seeds initial data."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Conversations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            message TEXT NOT NULL,
            intent TEXT NOT NULL,
            response TEXT NOT NULL,
            confidence REAL NOT NULL,
            model_used TEXT DEFAULT 'ml',
            is_emergency INTEGER DEFAULT 0,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    """)

    # Health Topics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS health_topics (
            topic_id INTEGER PRIMARY KEY,
            topic_name TEXT NOT NULL,
            description TEXT NOT NULL,
            prevention TEXT NOT NULL,
            when_to_seek_help TEXT NOT NULL
        )
    """)

    # Seed default admin if not exists
    cursor.execute("SELECT user_id FROM users WHERE email = ?", ("admin@healthbot.org",))
    if not cursor.fetchone():
        cursor.execute("""
            INSERT INTO users (name, email, password, role)
            VALUES (?, ?, ?, ?)
        """, ("Public Health Admin", "admin@healthbot.org", hash_password("admin123"), "admin"))

    # Seed default guest user if not exists
    cursor.execute("SELECT user_id FROM users WHERE email = ?", ("guest@healthbot.org",))
    if not cursor.fetchone():
        cursor.execute("""
            INSERT INTO users (name, email, password, role)
            VALUES (?, ?, ?, ?)
        """, ("Guest User", "guest@healthbot.org", hash_password("guest123"), "user"))

    conn.commit()
    conn.close()

    # Seed health topics from CSV if empty
    seed_health_topics_if_empty()

def seed_health_topics_if_empty():
    """Seeds health_topics table from data/health_topics.csv if empty."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as cnt FROM health_topics")
    count = cursor.fetchone()["cnt"]
    
    if count == 0:
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "health_topics.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                cursor.execute("""
                    INSERT OR REPLACE INTO health_topics (topic_id, topic_name, description, prevention, when_to_seek_help)
                    VALUES (?, ?, ?, ?, ?)
                """, (int(row["topic_id"]), row["topic_name"], row["description"], row["prevention"], row["when_to_seek_help"]))
            conn.commit()
            print(f"Seeded {len(df)} health topics into SQLite database.")
    conn.close()

# User Management
def register_user(name: str, email: str, password: str) -> dict:
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO users (name, email, password)
            VALUES (?, ?, ?)
        """, (name, email.strip().lower(), hash_password(password)))
        conn.commit()
        user_id = cursor.lastrowid
        return {"success": True, "user_id": user_id, "name": name, "email": email}
    except sqlite3.IntegrityError:
        return {"success": False, "error": "Email is already registered."}
    finally:
        conn.close()

def authenticate_user(email: str, password: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT user_id, name, email, role FROM users
        WHERE email = ? AND password = ?
    """, (email.strip().lower(), hash_password(password)))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

# Conversation Logging
def log_conversation(user_id: int, message: str, intent: str, response: str, confidence: float, model_used: str = "ml", is_emergency: bool = False):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO conversations (user_id, message, intent, response, confidence, model_used, is_emergency)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (user_id, message, intent, response, float(confidence), model_used, 1 if is_emergency else 0))
    conn.commit()
    conv_id = cursor.lastrowid
    conn.close()
    return conv_id

def get_user_conversations(user_id: int, limit: int = 50):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT conversation_id, message, intent, response, confidence, model_used, is_emergency, timestamp
        FROM conversations
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, (user_id, limit))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

# Health Topics
def get_all_health_topics():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM health_topics ORDER BY topic_id ASC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_health_topic_by_name(name_query: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM health_topics
        WHERE topic_name LIKE ? OR description LIKE ?
        LIMIT 5
    """, (f"%{name_query}%", f"%{name_query}%"))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

# Admin Dashboard Analytics
def get_admin_dashboard_stats():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Total users
    cursor.execute("SELECT COUNT(*) as count FROM users")
    total_users = cursor.fetchone()["count"]

    # Total questions / conversations
    cursor.execute("SELECT COUNT(*) as count FROM conversations")
    total_questions = cursor.fetchone()["count"]

    # Average confidence
    cursor.execute("SELECT AVG(confidence) as avg_conf FROM conversations")
    row_avg = cursor.fetchone()
    avg_confidence = round(row_avg["avg_conf"] or 0.0, 2)

    # Emergency triggers count
    cursor.execute("SELECT COUNT(*) as count FROM conversations WHERE is_emergency = 1")
    total_emergencies = cursor.fetchone()["count"]

    # Intent distribution (Top intents)
    cursor.execute("""
        SELECT intent, COUNT(*) as count
        FROM conversations
        GROUP BY intent
        ORDER BY count DESC
        LIMIT 10
    """)
    intent_distribution = [dict(r) for r in cursor.fetchall()]

    # Low-confidence queries (< 0.60)
    cursor.execute("""
        SELECT conversation_id, message, intent, confidence, model_used, timestamp
        FROM conversations
        WHERE confidence < 0.60
        ORDER BY timestamp DESC
        LIMIT 10
    """)
    low_confidence_queries = [dict(r) for r in cursor.fetchall()]

    # Daily chatbot usage (last 7 days)
    cursor.execute("""
        SELECT DATE(timestamp) as date, COUNT(*) as count
        FROM conversations
        GROUP BY DATE(timestamp)
        ORDER BY date DESC
        LIMIT 7
    """)
    daily_usage = [dict(r) for r in cursor.fetchall()]

    # Model usage distribution
    cursor.execute("""
        SELECT model_used, COUNT(*) as count
        FROM conversations
        GROUP BY model_used
    """)
    model_usage = [dict(r) for r in cursor.fetchall()]

    # Recent conversation logs (last 50)
    cursor.execute("""
        SELECT c.conversation_id, c.user_id, u.name as user_name, c.message, c.intent, c.response, c.confidence, c.model_used, c.is_emergency, c.timestamp
        FROM conversations c
        LEFT JOIN users u ON c.user_id = u.user_id
        ORDER BY c.timestamp DESC
        LIMIT 50
    """)
    recent_conversations = [dict(r) for r in cursor.fetchall()]

    # All registered users
    cursor.execute("SELECT user_id, name, email, role, created_at FROM users ORDER BY created_at DESC")
    all_users = [dict(r) for r in cursor.fetchall()]

    conn.close()
    return {
        "total_users": total_users,
        "total_questions": total_questions,
        "avg_confidence": avg_confidence,
        "total_emergencies": total_emergencies,
        "intent_distribution": intent_distribution,
        "low_confidence_queries": low_confidence_queries,
        "daily_usage": daily_usage,
        "model_usage": model_usage,
        "recent_conversations": recent_conversations,
        "all_users": all_users
    }

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")
