"""
src/auth.py

Authentication and Session Management Foundation for AI-Powered Customer Support Assistant.
Localhost Proof of Concept - Phase 1 Foundation.

Provides:
- In-memory demo user registry with salted PBKDF2-HMAC-SHA256 password hashes.
- User authentication with timing-safe comparison.
- In-memory session management with unpredictable session tokens.
- Session expiration validation and automated cleanup.
"""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import hashlib
import hmac
import secrets
import threading
from typing import Dict, Any, Optional

# Central configuration constants
SESSION_COOKIE_NAME = "session_id"
DEFAULT_SESSION_LIFETIME_SECONDS = 8 * 3600  # 8 hours
PBKDF2_ITERATIONS = 100_000
HASH_ALGORITHM = "sha256"


@dataclass(frozen=True)
class Session:
    """Represents an active authenticated support agent session."""
    session_id: str
    username: str
    agent_name: str
    created_at: datetime
    expires_at: datetime
    email: str = ""

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        """Helper method to check if this session instance is expired."""
        current_time = now if now is not None else datetime.now(timezone.utc)
        return self.expires_at <= current_time


# ------------------------------------------------------------------------------
# In-Memory Demo User Registry
# Plaintext passwords are NEVER stored in this registry.
# Passwords for demonstration:
#   - boomika (boomika@support.industrial.ai) : "Industrial@2026"
#   - alex    (alex@support.industrial.ai)    : "Machinery#400"
# ------------------------------------------------------------------------------
USER_REGISTRY: Dict[str, Dict[str, Any]] = {
    "boomika": {
        "username": "boomika",
        "email": "boomika@support.industrial.ai",
        "agent_name": "Boomika",
        "salt": bytes.fromhex("d8e0509387cc4abca615030b5bc3ec9c"),
        "password_hash": "02d1894f901c6168af0e5ad90d1d8a5e495572d34fe24a4648374c073117b647",
    },
    "alex": {
        "username": "alex",
        "email": "alex@support.industrial.ai",
        "agent_name": "Alex Chen",
        "salt": bytes.fromhex("3b1f5338bff24e1190ec4fb32c3327b1"),
        "password_hash": "e4eb3432a308b2d06d1bbf5200cd4241f06533f26bb35c1918300d40ff4fbba9",
    },
}

# In-memory session store: maps session_id -> Session object
_SESSIONS: Dict[str, Session] = {}
_session_lock = threading.Lock()


# ------------------------------------------------------------------------------
# Password Hashing & Verification
# ------------------------------------------------------------------------------
def hash_password(password: str, salt: bytes, iterations: int = PBKDF2_ITERATIONS) -> str:
    """
    Computes a PBKDF2-HMAC-SHA256 hex digest for a password and salt.
    """
    if not isinstance(password, str) or not password:
        raise ValueError("Password must be a non-empty string.")
    if not isinstance(salt, (bytes, bytearray)) or not salt:
        raise ValueError("Salt must be non-empty bytes.")
    derived = hashlib.pbkdf2_hmac(
        HASH_ALGORITHM,
        password.encode("utf-8"),
        salt,
        iterations
    )
    return derived.hex()


def verify_password(password: str, salt: bytes, expected_hash: str) -> bool:
    """
    Verifies a password against an expected PBKDF2 hash using timing-safe comparison.
    """
    if not password or not salt or not expected_hash:
        return False
    try:
        computed_hash = hash_password(password, salt)
        return hmac.compare_digest(computed_hash, expected_hash)
    except Exception:
        return False


# ------------------------------------------------------------------------------
# User Authentication
# ------------------------------------------------------------------------------
import re

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
_registry_lock = threading.Lock()


def get_user(identifier: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves user record from the registry by username or email (case-insensitive).
    """
    if not identifier or not isinstance(identifier, str):
        return None
    normalized = identifier.strip().lower()
    with _registry_lock:
        if normalized in USER_REGISTRY:
            return USER_REGISTRY[normalized]
        for user in USER_REGISTRY.values():
            if user.get("email", "").lower() == normalized:
                return user
    return None


def authenticate_user(username: str, password: str) -> Optional[Dict[str, str]]:
    """
    Authenticates an agent using username or email and password.

    Returns:
        dict with {"username": str, "agent_name": str} if credentials are valid,
        None if authentication fails.
    """
    if not username or not password:
        return None
    if not isinstance(username, str) or not isinstance(password, str):
        return None

    user = get_user(username)
    if not user:
        return None

    salt = user.get("salt")
    expected_hash = user.get("password_hash")
    if not salt or not expected_hash:
        return None

    if verify_password(password, salt, expected_hash):
        return {
            "username": user["username"],
            "agent_name": user["agent_name"],
        }
    return None


def register_user(agent_name: str, email: str, password: str) -> Dict[str, Any]:
    """
    Registers a new support agent in the in-memory user registry.

    NOTE: As this application is a localhost POC, newly registered accounts are held
    in-memory and will not survive a server restart. Existing demo accounts are statically seeded.

    Validation:
    - All fields required and non-empty.
    - Valid email format.
    - Email normalized (trimmed and lowercased).
    - Password minimum length >= 10 characters.
    - Rejects duplicate email addresses (case-insensitive).
    - Salt is securely generated with secrets.token_bytes(16).
    - Plaintext password is NEVER stored.

    Returns:
        dict with {"username": str, "email": str, "agent_name": str}
    Raises:
        ValueError if validation fails or email already exists.
    """
    if not agent_name or not isinstance(agent_name, str) or not agent_name.strip():
        raise ValueError("Agent full name is required.")
    if not email or not isinstance(email, str) or not email.strip():
        raise ValueError("Work email is required.")
    if not password or not isinstance(password, str):
        raise ValueError("Password is required.")

    clean_name = agent_name.strip()
    clean_email = email.strip().lower()

    if not EMAIL_REGEX.match(clean_email):
        raise ValueError("Please provide a valid work email address.")

    if len(password) < 10:
        raise ValueError("Password must be at least 10 characters long.")

    with _registry_lock:
        # Check for existing email (case-insensitive)
        for u in USER_REGISTRY.values():
            if u.get("email", "").lower() == clean_email:
                raise ValueError(f"An account with email '{clean_email}' already exists.")

        # Create unique sanitized username
        base_username = clean_email.split("@")[0]
        sanitized = re.sub(r"[^a-z0-9_-]", "", base_username) or "agent"
        candidate = sanitized
        counter = 1
        while candidate in USER_REGISTRY:
            candidate = f"{sanitized}_{counter}"
            counter += 1

        salt = secrets.token_bytes(16)
        pwd_hash = hash_password(password, salt)

        user_record = {
            "username": candidate,
            "email": clean_email,
            "agent_name": clean_name,
            "salt": salt,
            "password_hash": pwd_hash,
        }
        USER_REGISTRY[candidate] = user_record

        return {
            "username": candidate,
            "email": clean_email,
            "agent_name": clean_name,
        }


def reset_demo_registry() -> None:
    """
    Utility helper for test isolation. Resets USER_REGISTRY back to the static demo accounts.
    """
    with _registry_lock:
        dynamic_keys = [k for k in USER_REGISTRY if k not in ("boomika", "alex")]
        for k in dynamic_keys:
            del USER_REGISTRY[k]


# ------------------------------------------------------------------------------
# Session Management
# ------------------------------------------------------------------------------
def create_session(username: str, lifetime_seconds: Optional[int] = None) -> Session:
    """
    Creates a new authenticated session for a valid user.

    Raises:
        ValueError if the user is unknown or invalid.
    """
    user = get_user(username)
    if not user:
        raise ValueError(f"Cannot create session for unknown user '{username}'.")

    lifetime = lifetime_seconds if lifetime_seconds is not None else DEFAULT_SESSION_LIFETIME_SECONDS
    if lifetime <= 0:
        raise ValueError("Session lifetime must be a positive number of seconds.")

    session_id = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=lifetime)

    session = Session(
        session_id=session_id,
        username=user["username"],
        agent_name=user["agent_name"],
        created_at=now,
        expires_at=expires_at,
        email=user.get("email", ""),
    )

    with _session_lock:
        _SESSIONS[session_id] = session

    return session


def is_session_expired(session: Session, now: Optional[datetime] = None) -> bool:
    """
    Validates if a session has expired.
    """
    if not isinstance(session, Session):
        return True
    current_time = now if now is not None else datetime.now(timezone.utc)
    return session.expires_at <= current_time


def get_session(session_id: str, now: Optional[datetime] = None) -> Optional[Session]:
    """
    Retrieves an active session by session_id.
    If the session exists but has expired, it is removed and None is returned.
    """
    if not session_id or not isinstance(session_id, str):
        return None

    with _session_lock:
        session = _SESSIONS.get(session_id)
        if session is None:
            return None

        if is_session_expired(session, now=now):
            del _SESSIONS[session_id]
            return None

        return session


def delete_session(session_id: str) -> bool:
    """
    Removes a session by session_id.
    Returns True if the session was found and removed, False otherwise.
    """
    if not session_id or not isinstance(session_id, str):
        return False

    with _session_lock:
        return _SESSIONS.pop(session_id, None) is not None


def cleanup_expired_sessions(now: Optional[datetime] = None) -> int:
    """
    Purges all expired sessions from the in-memory store.
    Returns the number of removed sessions.
    """
    current_time = now if now is not None else datetime.now(timezone.utc)
    purged_count = 0

    with _session_lock:
        expired_ids = [
            sid for sid, s in _SESSIONS.items()
            if s.expires_at <= current_time
        ]
        for sid in expired_ids:
            del _SESSIONS[sid]
            purged_count += 1

    return purged_count


def clear_all_sessions() -> None:
    """
    Utility helper for test isolation. Clears all active sessions.
    """
    with _session_lock:
        _SESSIONS.clear()
