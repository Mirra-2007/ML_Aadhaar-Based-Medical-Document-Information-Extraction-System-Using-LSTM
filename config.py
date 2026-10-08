import os
import secrets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

DOCTORS_FILE = os.path.join(DATA_DIR, "doctors.json")
PATIENT_INDEX_FILE = os.path.join(DATA_DIR, "patient_index.json")
MEDICAL_DOCUMENTS_FILE = os.path.join(BASE_DIR, "sample_medical_documents.py")
FRONTEND_FILE = os.path.join(BASE_DIR, "frontend.html")
API_JS_FILE = os.path.join(BASE_DIR, "api.js")

# NLP preprocessing artifacts (Phase 4): vocabulary + encoded sequences
NLP_DIR = os.path.join(DATA_DIR, "nlp")
NLP_VOCAB_FILE = os.path.join(NLP_DIR, "vocab.json")
NLP_SEQUENCES_FILE = os.path.join(NLP_DIR, "sequences.json")
NLP_STATS_FILE = os.path.join(NLP_DIR, "preprocess_stats.json")

# Trained model artifacts (Phase 5): weights, config, vocab copy, tags, metrics
MODEL_DIR = os.path.join(BASE_DIR, "model")

_SECRET_KEY_FILE = os.path.join(DATA_DIR, ".secret_key")


def get_secret_key():
    """Resolve the Flask session signing key.

    Priority:
      1. FLASK_SECRET_KEY environment variable (production/deployed use)
      2. Persisted random key in data/.secret_key (generated on first run)
    The key is only ever read server-side; it is never sent to the frontend.
    """
    env_key = os.environ.get("FLASK_SECRET_KEY")
    if env_key:
        return env_key

    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(_SECRET_KEY_FILE):
        with open(_SECRET_KEY_FILE, "r", encoding="utf-8") as fh:
            key = fh.read().strip()
            if key:
                return key

    key = secrets.token_hex(32)
    with open(_SECRET_KEY_FILE, "w", encoding="utf-8") as fh:
        fh.write(key)
    try:
        os.chmod(_SECRET_KEY_FILE, 0o600)
    except OSError:
        pass
    return key
