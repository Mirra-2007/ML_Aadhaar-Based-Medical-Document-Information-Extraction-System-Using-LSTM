"""Patient lookup service for the Bubble Health demo portal.

Loads the synthetic medical document dataset and the synthetic
Aadhaar-like identifier index (data/patient_index.json), validates the
requested identifier, and returns a clean structured summary of the
patient's record.

Synthetic/demo data only. The identifiers below are fabricated
12-digit values created for this college project - they are NOT real
Aadhaar numbers and must never be replaced with real ones.
"""

import json
import os
import re
import threading

import config

# Exactly 12 numeric digits, nothing else.
AADHAAR_PATTERN = re.compile(r"^\d{12}$")

_lock = threading.Lock()
_documents_cache = None
_index_cache = None


class DatasetError(Exception):
    """Raised when the synthetic dataset or identifier index is unusable."""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_documents():
    global _documents_cache
    if _documents_cache is not None:
        return _documents_cache

    path = config.MEDICAL_DOCUMENTS_FILE
    if not os.path.exists(path):
        raise DatasetError("Medical dataset file is missing.")

    try:
        # The dataset file contains a JSON array despite the .py extension.
        with open(path, "r", encoding="utf-8") as fh:
            documents = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        raise DatasetError(f"Medical dataset could not be read: {exc}") from exc

    if not isinstance(documents, list) or not documents:
        raise DatasetError("Medical dataset is empty or malformed.")

    _documents_cache = {doc.get("document_id"): doc for doc in documents}
    return _documents_cache


def _load_index():
    global _index_cache
    if _index_cache is not None:
        return _index_cache

    path = config.PATIENT_INDEX_FILE
    if not os.path.exists(path):
        raise DatasetError("Synthetic patient identifier index is missing.")

    try:
        with open(path, "r", encoding="utf-8") as fh:
            index = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        raise DatasetError(f"Patient identifier index could not be read: {exc}") from exc

    if not isinstance(index, dict) or not index:
        raise DatasetError("Patient identifier index is empty or malformed.")

    _index_cache = index
    return _index_cache


# ---------------------------------------------------------------------------
# Extraction (same field patterns already used by SampleOutput.py)
# ---------------------------------------------------------------------------

def _extract(lines, pattern, default=None):
    regex = re.compile(pattern)
    for line in lines:
        match = regex.search(line)
        if match:
            return match.group(1).strip()
    return default


def _build_patient(identifier, document):
    """Map one raw dataset document to the clean dashboard response."""
    lines = document.get("parsed_lines", [])
    metadata = document.get("metadata", {})

    medications = []
    rx_pattern = re.compile(r"\*\s*Rx:\s*(.+)")
    for line in lines:
        match = rx_pattern.search(line)
        if match:
            medications.append(match.group(1).strip())

    return {
        "patient_id": identifier,
        "name": metadata.get("patient", "N/A"),
        "document_id": document.get("document_id", "N/A"),
        "encounter_date": metadata.get("date")
            or _extract(lines, r"DATE OF ENCOUNTER:\s*(.+)", "N/A"),
        "physician": metadata.get("physician", "N/A"),
        "chief_complaint": _extract(lines, r"Chief Complaint:\s*(.+)", "N/A"),
        "blood_pressure": _extract(lines, r"BP:\s*([\d/]+)", "N/A"),
        "heart_rate": _extract(lines, r"HR:\s*(\d+\s*bpm)", "N/A"),
        "glucose": _extract(lines, r"Blood Glucose:\s*([\d]+\s*mg/dL)", "N/A"),
        "cholesterol": _extract(lines, r"Cholesterol Total:\s*([\d]+\s*mg/dL)", "N/A"),
        "diagnosis": _extract(lines, r"Primary Assessment / Diagnosis:\s*(.+)", "N/A"),
        "medications": medications,
    }


# ---------------------------------------------------------------------------
# Resolve (shared by lookup and LSTM extraction)
# ---------------------------------------------------------------------------

def resolve_document(identifier):
    """Resolve a synthetic Aadhaar identifier to a raw dataset document plus
    the structured patient summary.

    Returns (status_code, payload_dict). 200 payload carries:
        patient_id, document_id,
        patient  (same clean schema as lookup_patient),
        document (raw dataset record, including parsed_lines).

    Mirrors the old lookup_patient semantics exactly (same 400/404/500 cases)
    so the LSTM extraction endpoint can validate/retrieve in one step. The
    protected /api/patient/lookup endpoint keeps its exact response shape.
    """
    if not isinstance(identifier, str) or not AADHAAR_PATTERN.match(identifier.strip()):
        return 400, {
            "status": "error",
            "message": "Invalid identifier format. Enter exactly 12 digits.",
        }

    identifier = identifier.strip()

    try:
        index = _load_index()
        documents = _load_documents()
    except DatasetError as exc:
        return 500, {"status": "error", "message": str(exc)}

    document_id = index.get(identifier)
    if not document_id:
        # Unknown identifier: reveal nothing about the dataset.
        return 404, {
            "status": "error",
            "message": "No patient record found.",
        }

    document = documents.get(document_id)
    if not document:
        return 404, {
            "status": "error",
            "message": "No patient record found.",
        }

    try:
        patient = _build_patient(identifier, document)
    except Exception as exc:  # noqa: BLE001 - surface as a clean 500
        return 500, {
            "status": "error",
            "message": f"Failed to process the patient document: {exc}",
        }

    return 200, {
        "status": "success",
        "patient_id": identifier,
        "document_id": document_id,
        "patient": patient,
        "document": document,
    }


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

def lookup_patient(identifier):
    """Resolve a synthetic Aadhaar identifier to structured patient data.

    Returns (status_code, payload_dict) with the same response shape as
    before Phase 6 (the raw document is stripped before returning).
    """
    status, payload = resolve_document(identifier)
    if status != 200:
        return status, payload
    return 200, {"status": "success", "patient": payload["patient"]}
