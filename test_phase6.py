"""Phase 6 tests: full authenticated LSTM-extraction path + error cases.

Coverage (Phase 6 requirements):
  * E2E: signup -> sign in -> valid ID -> document retrieval -> NLP -> saved
    LSTM inference -> structured summary -> logout -> extraction 401.
  * wrong ID -> 400; unknown ID -> 404; unauthenticated -> 401.
  * model artifacts missing -> 503 (no fake/regex fallback claim).
  * repeated extraction confirms the model is never retrained (weights mtime
    unchanged) and the singleton is reused.
  * /api/patient/lookup regression stays intact.
Requires the trained model/ artifacts (created by `py train.py`).
"""

import os
import tempfile
import unittest

import config
import inference
import server
from patient_service import AADHAAR_PATTERN


class Phase6ExtractionTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = server.app.test_client()

    def setUp(self):
        # Load the singleton fresh per test so the 503 scenario (which points
        # MODEL_DIR elsewhere) can never leak a stale in-memory model. In real
        # app flow the singleton stays loaded per process (see no-retrain test).
        inference.reset_extractor_for_tests()

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _signup_or_login(client, username="DR-9913"):
        """Create the demo doctor (or just sign in if already registered)."""
        body = {
            "username": username,
            "name": "Phase6 Test Doctor",
            "password": "TestPass1",
            "confirm_password": "TestPass1",
            "specialization": "General Medicine",
        }
        response = client.post("/api/auth/signup", json=body)
        if response.status_code not in (201, 409):
            return response
        return client.post(
            "/api/auth/login", json={"username": username, "password": "TestPass1"}
        )

    def _authenticated_client(self):
        client = server.app.test_client()
        response = self._signup_or_login(client)
        self.assertEqual(
            response.status_code, 200,
            f"signup/login failed: {response.get_json()}",
        )
        return client

    @staticmethod
    def _norm(text):
        return " ".join((text or "").split())

    # ------------------------------------------------------------------- E2E

    def test_01_e2e_signup_login_extract_dashboard_logout_401(self):
        client = self._authenticated_client()

        response = client.post("/api/patient/extract", json={"aadhaar": "123456789012"})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["document_id"], "MED-1")
        self.assertEqual(data["patient_id"], "123456789012")

        source = data["source_dataset"]
        self.assertEqual(source["document_id"], "MED-1")
        self.assertEqual(source["blood_pressure"], "127/82")

        extracted = data["lstm_extracted"]
        self.assertTrue(extracted["entities"], "LSTM returned no entities")
        entity_types = {entity["type"] for entity in extracted["entities"]}
        self.assertTrue(entity_types.issubset({"BP", "HR", "GLU", "CHOL", "DX", "MED", "DOSE"}))
        self.assertEqual(data["extraction"]["method"], "saved-lstm-no-retrain")

        # LSTM-extracted values agree with the dataset's own fields (MED-1).
        by_type = {}
        for entity in extracted["entities"]:
            by_type.setdefault(entity["type"], []).append(self._norm(entity["text"]))
        self.assertIn(self._norm("127/82"), by_type.get("BP", []))
        self.assertIn(self._norm(source["heart_rate"]), by_type.get("HR", []))
        self.assertIn(self._norm(source["glucose"]), by_type.get("GLU", []))
        self.assertIn(self._norm(source["cholesterol"]), by_type.get("CHOL", []))
        # The diagnosis line ends with "." but the tokenizer splits it off the
        # LSTM's DX span, so compare with trailing punctuation stripped.
        dx_candidates = [text.rstrip(".") for text in by_type.get("DX", [])]
        self.assertIn(self._norm(source["diagnosis"]).rstrip("."), dx_candidates)

        logout = client.post("/api/auth/logout")
        self.assertEqual(logout.status_code, 200)

        after_logout = client.post("/api/patient/extract", json={"aadhaar": "123456789012"})
        self.assertEqual(after_logout.status_code, 401)

    def test_02_wrong_id_400(self):
        client = self._authenticated_client()
        for bad in ("12345", "abcdabcdabcd", "12345678901a", ""):
            response = client.post("/api/patient/extract", json={"aadhaar": bad})
            self.assertEqual(response.status_code, 400, f"id={bad!r}")
            self.assertFalse(AADHAAR_PATTERN.match(bad or ""), f"id={bad!r} should be invalid")

    def test_03_unknown_id_404(self):
        client = self._authenticated_client()
        response = client.post("/api/patient/extract", json={"aadhaar": "999999999999"})
        self.assertEqual(response.status_code, 404)

    def test_04_unauthenticated_401(self):
        fresh = server.app.test_client()
        response = fresh.post("/api/patient/extract", json={"aadhaar": "123456789012"})
        self.assertEqual(response.status_code, 401)

    def test_05_model_artifacts_missing_503(self):
        client = self._authenticated_client()
        original_dir = config.MODEL_DIR
        with tempfile.TemporaryDirectory() as empty_dir:
            config.MODEL_DIR = empty_dir
            inference.reset_extractor_for_tests()
            try:
                response = client.post("/api/patient/extract", json={"aadhaar": "123456789012"})
            finally:
                config.MODEL_DIR = original_dir
                inference.reset_extractor_for_tests()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()["code"], "model_unavailable")

    def test_06_repeated_extraction_no_retrain_singleton(self):
        client = self._authenticated_client()
        weights_path = os.path.join(original := config.MODEL_DIR, "model_weights.pt")
        self.assertTrue(os.path.exists(weights_path), "trained artifacts missing")

        before = os.path.getmtime(weights_path)
        first = client.post("/api/patient/extract", json={"aadhaar": "123456789012"})
        second = client.post("/api/patient/extract", json={"aadhaar": "987654321098"})
        third = client.post("/api/patient/extract", json={"aadhaar": "100000000003"})
        after = os.path.getmtime(weights_path)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(third.status_code, 200)

        # The weights file was never rewritten -> no retraining took place.
        self.assertEqual(before, after)
        # The process-wide singleton is reused for every call.
        self.assertIs(inference.get_extractor(), inference.get_extractor())
        self.assertEqual(second.get_json()["document_id"], "MED-2")
        self.assertEqual(third.get_json()["document_id"], "MED-3")

    def test_07_lookup_regression_unchanged(self):
        client = self._authenticated_client()
        response = client.post("/api/patient/lookup", json={"aadhaar": "123456789012"})
        self.assertEqual(response.status_code, 200)
        patient = response.get_json()["patient"]
        self.assertEqual(patient["document_id"], "MED-1")
        self.assertEqual(patient["blood_pressure"], "127/82")
        # The lookup response must not carry the LSTM raw fields.
        self.assertNotIn("lstm_extracted", response.get_json())

    def test_08_clean_response_separates_source_and_lstm_fields(self):
        client = self._authenticated_client()
        data = client.post(
            "/api/patient/extract", json={"aadhaar": "123456789012"}
        ).get_json()
        # Requirement: source dataset fields vs NLP/LSTM-extracted fields must be
        # clearly distinguishable (different top-level keys + provenance note).
        self.assertIn("source_dataset", data)
        self.assertIn("lstm_extracted", data)
        self.assertEqual(data["extraction"]["method"], "saved-lstm-no-retrain")
        self.assertIn("no retraining", data["extraction"]["provenance"].lower())


if __name__ == "__main__":
    unittest.main()