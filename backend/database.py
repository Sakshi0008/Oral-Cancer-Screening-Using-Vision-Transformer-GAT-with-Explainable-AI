"""
ORALSCAN AI - Database Management Layer
======================================
Secure SQLite database interface handling user authentication, session state,
and screening records with strict user isolation.
"""

import os
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path


DEFAULT_DB_PATH = Path("data/oralscan.db")


class DatabaseManager:
    """
    Thread-safe SQLite database manager for OralScan AI.
    """
    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        """Initializes tables for users, sessions, and screenings."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS screenings (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    original_filename TEXT NOT NULL,
                    stored_image_path TEXT NOT NULL,
                    pred_label INTEGER NOT NULL,
                    pred_name TEXT NOT NULL,
                    cautious_result TEXT NOT NULL,
                    cancer_score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    vit_share_pct REAL NOT NULL,
                    gat_share_pct REAL NOT NULL,
                    ig_abs_error REAL NOT NULL,
                    ig_rel_error_pct REAL NOT NULL,
                    narrative TEXT NOT NULL,
                    multipanel_viz_path TEXT,
                    rollout_viz_path TEXT,
                    graph_viz_path TEXT,
                    pdf_report_path TEXT,
                    has_perturbation INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_screenings_user ON screenings(user_id);
                CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token);
            """)

    @staticmethod
    def _hash_password(password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
        """Hashes password using PBKDF2-HMAC-SHA256 with 100,000 iterations."""
        if salt is None:
            salt = secrets.token_bytes(16)
        pwd_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            100000
        )
        return pwd_hash.hex(), salt.hex()

    def register_user(self, username: str, email: str, password: str) -> Dict[str, Any]:
        """Registers a new user with secure password hashing."""
        username = username.strip().lower()
        email = email.strip().lower()
        if len(username) < 3:
            raise ValueError("Username must be at least 3 characters long.")
        if len(password) < 6:
            raise ValueError("Password must be at least 6 characters long.")

        pwd_hash, salt_hex = self._hash_password(password)
        try:
            with self._get_connection() as conn:
                cur = conn.execute(
                    "INSERT INTO users (username, email, password_hash, salt) VALUES (?, ?, ?, ?)",
                    (username, email, pwd_hash, salt_hex)
                )
                user_id = cur.lastrowid
                return {"id": user_id, "username": username, "email": email}
        except sqlite3.IntegrityError as e:
            if "username" in str(e):
                raise ValueError("Username is already registered.")
            elif "email" in str(e):
                raise ValueError("Email address is already registered.")
            raise ValueError("User registration failed due to existing record.")

    def authenticate_user(self, username_or_email: str, password: str) -> Optional[Dict[str, Any]]:
        """Verifies credentials and returns user dictionary if valid."""
        identifier = username_or_email.strip().lower()
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT id, username, email, password_hash, salt FROM users WHERE username = ? OR email = ?",
                (identifier, identifier)
            )
            row = cur.fetchone()
            if not row:
                return None

            expected_hash = row["password_hash"]
            salt = bytes.fromhex(row["salt"])
            computed_hash, _ = self._hash_password(password, salt=salt)

            if secrets.compare_digest(expected_hash, computed_hash):
                return {"id": row["id"], "username": row["username"], "email": row["email"]}
            return None

    def create_session(self, user_id: int, duration_days: int = 7) -> str:
        """Generates a secure 32-byte session token."""
        token = secrets.token_hex(32)
        expires_at = datetime.utcnow() + timedelta(days=duration_days)
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
                (token, user_id, expires_at.isoformat())
            )
        return token

    def get_user_by_session(self, token: str) -> Optional[Dict[str, Any]]:
        """Validates session token and returns active user."""
        now = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            cur = conn.execute("""
                SELECT u.id, u.username, u.email
                FROM sessions s
                JOIN users u ON s.user_id = u.id
                WHERE s.token = ? AND s.expires_at > ?
            """, (token, now))
            row = cur.fetchone()
            if row:
                return {"id": row["id"], "username": row["username"], "email": row["email"]}
            return None

    def delete_session(self, token: str) -> None:
        """Deletes session on logout."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))

    def save_screening(
        self,
        screening_id: str,
        user_id: int,
        original_filename: str,
        stored_image_path: str,
        pred_label: int,
        pred_name: str,
        cautious_result: str,
        cancer_score: float,
        confidence: float,
        vit_share_pct: float,
        gat_share_pct: float,
        ig_abs_error: float,
        ig_rel_error_pct: float,
        narrative: str,
        multipanel_viz_path: Optional[str] = None,
        rollout_viz_path: Optional[str] = None,
        graph_viz_path: Optional[str] = None,
        pdf_report_path: Optional[str] = None,
        has_perturbation: bool = False
    ) -> None:
        """Persists a verified screening outcome with ownership tag."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO screenings (
                    id, user_id, original_filename, stored_image_path,
                    pred_label, pred_name, cautious_result, cancer_score, confidence,
                    vit_share_pct, gat_share_pct, ig_abs_error, ig_rel_error_pct,
                    narrative, multipanel_viz_path, rollout_viz_path, graph_viz_path,
                    pdf_report_path, has_perturbation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                screening_id, user_id, original_filename, stored_image_path,
                pred_label, pred_name, cautious_result, cancer_score, confidence,
                vit_share_pct, gat_share_pct, ig_abs_error, ig_rel_error_pct,
                narrative, multipanel_viz_path, rollout_viz_path, graph_viz_path,
                pdf_report_path, 1 if has_perturbation else 0
            ))

    def update_pdf_path(self, screening_id: str, user_id: int, pdf_path: str) -> bool:
        """Updates the generated PDF path for a specific user screening."""
        with self._get_connection() as conn:
            cur = conn.execute(
                "UPDATE screenings SET pdf_report_path = ? WHERE id = ? AND user_id = ?",
                (pdf_path, screening_id, user_id)
            )
            return cur.rowcount > 0

    def get_screening(self, screening_id: str, user_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a screening record strictly enforcing user ownership."""
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM screenings WHERE id = ? AND user_id = ?",
                (screening_id, user_id)
            )
            row = cur.fetchone()
            if row:
                return dict(row)
            return None

    def list_user_screenings(self, user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """Lists user screening history sorted by date descending."""
        with self._get_connection() as conn:
            cur = conn.execute("""
                SELECT id, original_filename, pred_label, pred_name, cautious_result,
                       cancer_score, confidence, vit_share_pct, gat_share_pct,
                       multipanel_viz_path, pdf_report_path, created_at
                FROM screenings
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (user_id, limit))
            return [dict(r) for r in cur.fetchall()]

    def delete_screening(self, screening_id: str, user_id: int) -> bool:
        """Deletes a screening record if owned by user."""
        with self._get_connection() as conn:
            cur = conn.execute(
                "DELETE FROM screenings WHERE id = ? AND user_id = ?",
                (screening_id, user_id)
            )
            return cur.rowcount > 0

    def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """Calculates dashboard summary statistics."""
        with self._get_connection() as conn:
            cur = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN pred_label = 1 THEN 1 ELSE 0 END) as suspicious,
                    SUM(CASE WHEN pred_label = 0 THEN 1 ELSE 0 END) as benign
                FROM screenings
                WHERE user_id = ?
            """, (user_id,))
            row = cur.fetchone()
            return {
                "total_screenings": row["total"] or 0,
                "suspicious_count": row["suspicious"] or 0,
                "benign_count": row["benign"] or 0
            }
