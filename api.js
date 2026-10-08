/**
 * api.js - Single place for all frontend <-> Flask communication.
 *
 * Every page function calls these helpers; no raw fetch() calls live in the
 * HTML. All requests send cookies (credentials: "include") so the Flask
 * session cookie authenticates protected routes.
 *
 * Error convention: helpers throw an object { status, message } where
 * status === 0 means the backend could not be reached at all.
 */

const API_BASE_URL = "http://127.0.0.1:5000"; // Same origin as the page (Flask serves the frontend).

async function apiRequest(path, { method = "GET", body = null } = {}) {
  let response;
  try {
    response = await fetch(API_BASE_URL + path, {
      method,
      headers: body ? { "Content-Type": "application/json" } : {},
      body: body ? JSON.stringify(body) : null,
      credentials: "include",
    });
  } catch (err) {
    throw {
      status: 0,
      message:
        "Backend unavailable. Please make sure the Flask server is running (python server.py).",
    };
  }

  let data = null;
  try {
    data = await response.json();
  } catch (err) {
    // Non-JSON response body; fall through to status handling.
  }

  if (!response.ok) {
    throw {
      status: response.status,
      message: (data && data.message) || `Request failed with status ${response.status}.`,
    };
  }

  return data;
}

/* ------------------------------------------------------------------ */
/* Doctor authentication                                               */
/* ------------------------------------------------------------------ */

/**
 * Create a doctor account.
 * @param {{username:string, name:string, password:string, confirm_password:string, specialization?:string}} payload
 * @returns {Promise<object>} {status, message, doctor}
 */
function signupDoctor(payload) {
  return apiRequest("/api/auth/signup", { method: "POST", body: payload });
}

/**
 * Sign in an existing doctor (sets the session cookie on success).
 * @param {string} username
 * @param {string} password
 * @returns {Promise<object>} {status, message, doctor}
 */
function loginDoctor(username, password) {
  return apiRequest("/api/auth/login", {
    method: "POST",
    body: { username, password },
  });
}

/**
 * End the current doctor session.
 * @returns {Promise<object>} {status, message}
 */
function logoutDoctor() {
  return apiRequest("/api/auth/logout", { method: "POST" });
}

/**
 * Ask the backend whether the current session is authenticated.
 * Always resolves (never throws for a normal "not signed in" answer).
 * @returns {Promise<{authenticated:boolean, doctor:object|null}>}
 */
async function checkSession() {
  try {
    const data = await apiRequest("/api/auth/session");
    return { authenticated: !!data.authenticated, doctor: data.doctor || null };
  } catch (err) {
    // Backend down or network failure -> treat as signed out.
    return { authenticated: false, doctor: null, unreachable: err.status === 0 };
  }
}

/* ------------------------------------------------------------------ */
/* Patient lookup                                                      */
/* ------------------------------------------------------------------ */

/**
 * Look up a patient record by 12-digit synthetic Aadhaar identifier.
 * Requires an authenticated doctor session (backend enforces 401).
 * @param {string} aadhaar - exactly 12 digits, spaces already removed
 * @returns {Promise<{patient: object}>} {status, patient}
 */
function lookupPatient(aadhaar) {
  return apiRequest("/api/patient/lookup", {
    method: "POST",
    body: { aadhaar },
  });
}

/**
 * Run the full LSTM extraction flow for a patient record.
 * Authenticated. Combines the deterministic dataset fields
 * ({source_dataset}, same schema as lookup) with the entities produced
 * solely by the saved LSTM ({lstm_extracted}).
 * @param {string} aadhaar - exactly 12 digits, spaces already removed
 * @returns {Promise<{source_dataset: object, lstm_extracted: object}>}
 */
function extractPatient(aadhaar) {
  return apiRequest("/api/patient/extract", {
    method: "POST",
    body: { aadhaar },
  });
}
