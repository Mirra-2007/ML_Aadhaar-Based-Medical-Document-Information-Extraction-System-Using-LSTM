"""Doctor authentication for the Bubble Health demo portal.

Handles username/password validation, bcrypt password hashing, the local
doctor registry (data/doctors.json) and the Flask session guard used to
protect authenticated API routes.

Synthetic/demo use only - no real credentials or real patient data.
"""

import json
import os
import re
import threading
from datetime import datetime, timezone
from functools import wraps

import bcrypt
from flask import jsonify, session

import config

# Required project username pattern - do not change.
DOCTOR_USERNAME_PATTERN = re.compile(
    r"^(?:[a-zA-Z]{2,8}[/\s-]?\d{3,7}|\d{4,7}|[a-zA-Z0-9._-]+@hpr\.abdm)$"
)

PASSWORD_MIN_LENGTH = 8

_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def is_valid_username(username):
    """True when the username matches the required pattern exactly."""
    if not isinstance(username, str):
        return False
    return bool(DOCTOR_USERNAME_PATTERN.match(username))


def validate_password(password):
    """Return (ok, message). Checks length plus upper/lower/digit mix."""
    if not isinstance(password, str) or len(password) < PASSWORD_MIN_LENGTH:
        return False, (
            "Password must be at least "
            f"{PASSWORD_MIN_LENGTH} characters long."
        )
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter."
    if not re.search(r"\d", password):
        return False, "Password must contain at least one number."
    return True, ""


def validate_name(name):
    if not isinstance(name, str):
        return False
    cleaned = name.strip()
    return 1 <= len(cleaned) <= 100


# ---------------------------------------------------------------------------
# Doctor registry storage (JSON file, bcrypt hashes only - never plaintext)
# ---------------------------------------------------------------------------

def _load_registry():
    if not os.path.exists(config.DOCTORS_FILE):
        return {}
    try:
        with open(config.DOCTORS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_registry(registry):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    tmp_path = config.DOCTORS_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=4, ensure_ascii=False)
    os.replace(tmp_path, config.DOCTORS_FILE)


def _public_doctor(record):
    """Shape a stored record for API responses (never includes the hash)."""
    return {
        "username": record.get("username", ""),
        "name": record.get("name", ""),
        "specialization": record.get("specialization", ""),
    }


# ---------------------------------------------------------------------------
# Sign up / Sign in
# ---------------------------------------------------------------------------

def create_doctor(username, name, password, confirm_password, specialization=""):
    """Register a new doctor.

    Returns (status_code, payload_dict).
    """
    username = (username or "").strip()
    name = (name or "").strip()
    specialization = (specialization or "").strip()

    if not username:
        return 400, {"status": "error", "message": "Doctor username is required."}
    if not is_valid_username(username):
        return 400, {
            "status": "error",
            "message": (
                "Invalid doctor username format. Allowed: 2-8 letters + 3-7 "
                "digits (e.g. DR-1234), 4-7 digits, or name@hpr.abdm."
            ),
        }
    if not validate_name(name):
        return 400, {"status": "error", "message": "Doctor name must be 1-100 characters."}
    if not password or not confirm_password:
        return 400, {"status": "error", "message": "Password and confirmation are required."}
    if password != confirm_password:
        return 400, {"status": "error", "message": "Passwords do not match."}

    ok, message = validate_password(password)
    if not ok:
        return 400, {"status": "error", "message": message}

    key = username.lower()

    with _lock:
        registry = _load_registry()
        if key in registry:
            return 409, {
                "status": "error",
                "message": "This doctor username is already registered. Please sign in instead.",
            }

        password_hash = bcrypt.hashpw(
            password.encode("utf-8"), bcrypt.gensalt(rounds=12)
        ).decode("utf-8")

        registry[key] = {
            "username": username,
            "name": name,
            "specialization": specialization,
            "password_hash": password_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_registry(registry)

    return 201, {
        "status": "success",
        "message": "Account created successfully. You can now sign in.",
        "doctor": _public_doctor(registry[key]),
    }


def verify_credentials(username, password):
    """Check a sign-in attempt.

    Returns (status_code, payload_dict). Failure messages are deliberately
    identical for unknown username and wrong password.
    """
    username = (username or "").strip()
    if not username or not password:
        return 400, {"status": "error", "message": "Username and password are required."}

    registry = _load_registry()
    record = registry.get(username.lower())

    if not record:
        # Run a dummy hash so timing is roughly comparable for unknown users.
        bcrypt.checkpw(b"invalid", bcrypt.gensalt(rounds=12))
        return 401, {
            "status": "error",
            "message": "Invalid username or password. Please try again.",
        }

    try:
        match = bcrypt.checkpw(
            password.encode("utf-8"),
            record["password_hash"].encode("utf-8"),
        )
    except (ValueError, KeyError):
        match = False

    if not match:
        return 401, {
            "status": "error",
            "message": "Invalid username or password. Please try again.",
        }

    return 200, {
        "status": "success",
        "message": "Signed in successfully.",
        "doctor": _public_doctor(record),
    }


# ---------------------------------------------------------------------------
# Flask session guard
# ---------------------------------------------------------------------------

def login_user(username, name):
    """Establish the server-side session for an authenticated doctor."""
    session.clear()
    session["doctor_username"] = username
    session["doctor_name"] = name
    session.permanent = True


def current_doctor():
    """Public info for the signed-in doctor, or None."""
    username = session.get("doctor_username")
    if not username:
        return None
    record = _load_registry().get(username.lower())
    if not record:
        return None
    return _public_doctor(record)


def login_required(view_function):
    """Reject unauthenticated requests to protected API routes with 401 JSON."""
    @wraps(view_function)
    def wrapper(*args, **kwargs):
        if not session.get("doctor_username"):
            return jsonify({
                "status": "error",
                "code": "unauthenticated",
                "message": "Authentication required. Please sign in as a doctor.",
            }), 401
        return view_function(*args, **kwargs)

    return wrapper
