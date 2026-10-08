from datetime import timedelta

from flask import Flask, current_app, jsonify, request, send_file, session
from flask_cors import CORS
import json

import auth
import config
import inference
import patient_service

app = Flask(__name__)
app.secret_key = config.get_secret_key()
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,       # JavaScript cannot read the session cookie
    SESSION_COOKIE_SAMESITE="Lax",      # Sent on same-site requests only
    PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
)
CORS(app, supports_credentials=True)  # Enable CORS to allow requests from your frontend


# ---------------------------------------------------------------------------
# Static frontend serving (login flow uses these routes)
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_file(config.FRONTEND_FILE)


@app.route("/api.js")
def api_js():
    return send_file(config.API_JS_FILE)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.route("/api/health")
def health():
    return jsonify({"status": "success", "message": "API is healthy."}), 200


# ---------------------------------------------------------------------------
# Doctor authentication
# ---------------------------------------------------------------------------

@app.route("/api/auth/signup", methods=["POST"])
def signup():
    content = request.get_json(silent=True) or {}
    status, payload = auth.create_doctor(
        username=content.get("username"),
        name=content.get("name"),
        password=content.get("password"),
        confirm_password=content.get("confirm_password"),
        specialization=content.get("specialization", ""),
    )
    return jsonify(payload), status


@app.route("/api/auth/login", methods=["POST"])
def login():
    content = request.get_json(silent=True) or {}
    status, payload = auth.verify_credentials(
        username=content.get("username"),
        password=content.get("password"),
    )
    if status == 200:
        doctor = payload["doctor"]
        auth.login_user(doctor["username"], doctor["name"])
    return jsonify(payload), status


@app.route("/api/auth/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"status": "success", "message": "Signed out successfully."}), 200


@app.route("/api/auth/session")
def session_info():
    doctor = auth.current_doctor()
    if doctor is None:
        return jsonify({"status": "success", "authenticated": False, "doctor": None}), 200
    return jsonify({"status": "success", "authenticated": True, "doctor": doctor}), 200


# ---------------------------------------------------------------------------
# Patient lookup (protected - doctors only)
# ---------------------------------------------------------------------------

@app.route("/api/patient/lookup", methods=["POST"])
@auth.login_required
def patient_lookup():
    try:
        content = request.get_json(silent=True) or {}
        status, payload = patient_service.lookup_patient(content.get("aadhaar"))
        return jsonify(payload), status
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/patient/extract", methods=["POST"])
@auth.login_required
def patient_extract():
    """Authenticated LSTM extraction: ID -> document -> NLP -> LSTM -> summary.

    Response clearly separates the two field families (Phase 6 requirement):
      * source_dataset : deterministic values parsed from the dataset file
                         (same fields as /api/patient/lookup).
      * lstm_extracted : entities produced by the saved LSTM only
                         (inference.get_extractor over model/ artifacts).
    The LSTM is loaded lazily IF and ONLY IF the model/ artifacts exist, is
    loaded at most once per process, and is never retrained here or anywhere
    in this flow.
    """
    try:
        content = request.get_json(silent=True) or {}
        status, payload = patient_service.resolve_document(content.get("aadhaar"))
        if status != 200:
            return jsonify(payload), status

        try:
            extractor = inference.get_extractor()
        except Exception as exc:  # noqa: BLE001 - missing/corrupt artifacts -> 503
            current_app.logger.warning("LSTM artifacts unavailable: %s", exc)
            return jsonify({
                "status": "error",
                "code": "model_unavailable",
                "message": (
                    "LSTM model artifacts are unavailable. Run `py train.py` "
                    "to train and save the model before extracting."
                ),
            }), 503

        try:
            lstm_result = extractor.extract_lines(
                payload["document"].get("parsed_lines", [])
            )
        except Exception as exc:  # noqa: BLE001 - surface as a clean 500
            current_app.logger.error("LSTM extraction failed: %s", exc)
            return jsonify({"status": "error", "message": str(exc)}), 500

        return jsonify({
            "status": "success",
            "patient_id": payload["patient_id"],
            "document_id": payload["document_id"],
            "source_dataset": payload["patient"],
            "lstm_extracted": lstm_result,
            "extraction": {
                "method": "saved-lstm-no-retrain",
                "provenance": (
                    "lstm_extracted entities are produced only by the saved "
                    "LSTM (inference.get_extractor singleton over model/ "
                    "artifacts); source_dataset fields are deterministic "
                    "values parsed from the sample_medical_documents.py "
                    "dataset. No retraining occurs during requests."
                ),
            },
        }), 200
    except Exception as exc:  # noqa: BLE001 - surface as a clean 500
        return jsonify({"status": "error", "message": str(exc)}), 500


def extract_important_points(raw_dataset_text):
    """
    Simulates the extraction of important points from a raw unstructured 
    or structured patient dataset input. 
    In a production environment, you can integrate OpenAI, HuggingFace, 
    or regex patterns here.
    """
    # Example logic mapping extracted points to your frontend JSON schema
    # For demonstration, we parse or structure the incoming text data:
    
    # Placeholder for extracted structured data
    extracted_data = {
        "name": "Extracted Patient Profile",
        "gender": "Unknown",
        "bloodGroup": "N/A",
        "bp": "N/A",
        "diabetes": "N/A",
        "cholesterol": "N/A",
        "details": {
            "diabetesDesc": "Extracted metabolic status pending verification.",
            "bpDesc": "Extracted cardiovascular metrics pending verification.",
            "genetic": [
                f"Extracted point from dataset: {raw_dataset_text[:100]}..."
            ],
            "specialTrack": {
                "title": "Automated Clinical Summary",
                "icon": "activity",
                "html": f"""
                    <div class="space-y-2 text-sm text-slate-300">
                        <div><span class="text-xs text-goldBorder font-bold block uppercase mb-0.5">Dataset Source Analysis</span>Processed successfully from input record.</div>
                    </div>
                """
            }
        }
    }
    
    return extracted_data

@app.route('/api/process-dataset', methods=['POST'])
@auth.login_required
def process_dataset():
    try:
        content = request.json
        raw_dataset = content.get('dataset', '')

        if not raw_dataset:
            return jsonify({"error": "Dataset input is empty"}), 400

        # Step: Extract important points from the given dataset input
        important_points = extract_important_points(raw_dataset)

        # Return the structured important points as output
        return jsonify({
            "status": "success",
            "message": "Dataset processed and important points extracted successfully.",
            "output": important_points
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
