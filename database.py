import sqlite3
import os
import hashlib

DB_PATH = os.path.join(os.path.dirname(__file__), 'users.db')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row # rows behave like dicts
    return conn


def init_db():
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                email         TEXT    NOT NULL UNIQUE,
                password_hash TEXT    NOT NULL,
                created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def create_user(email: str, password: str) -> bool:
    try:
        with get_db() as conn:
            conn.execute('INSERT INTO users (email, password_hash) VALUES (?, ?)', (email, hash_password(password)))
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # Email already exists
    

def get_user_by_email(email: str):
    with get_db() as conn:
        return conn.execute(
            'SELECT * FROM users WHERE email = ?',
            (email.lower().strip(),)
        ).fetchone()
    

def verify_password(email: str, password: str) -> bool:
    user = get_user_by_email(email)
    if user is None:
        return False
    return user['password_hash'] == hash_password(password)