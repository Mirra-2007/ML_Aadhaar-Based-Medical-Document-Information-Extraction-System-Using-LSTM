"""Phase 5 tests: preprocessing -> LSTM -> prediction -> save -> reload.

Run from the project root (after `py train.py` has produced model/):

    py -m unittest test_lstm -v

Covers the required smoke chain plus gate correctness, label generation
consistency, BIO validity, and the no-retrain/load-once inference rules.
"""

import inspect
import json
import os
import unittest

import torch
import torch.nn as nn

import config
import data_prep
import inference
import labels
import lstm_model
import nlp_preprocess


def _load_model():
    return inference.Extractor.load()


class TestCasePreprocessingToSequence(unittest.TestCase):
    """Requirement 17: preprocessing -> sequence input."""

    @classmethod
    def setUpClass(cls):
        cls.samples = data_prep.build_samples()
        cls.vocab = nlp_preprocess.load_vocab()

    def test_tokens_encode_to_ids_aligned(self):
        sample = self.samples[0]
        self.assertEqual(len(sample["tokens"]), len(sample["token_ids"]))
        self.assertEqual(len(sample["tokens"]), len(sample["tags"]))
        self.assertTrue(all(0 <= i < len(self.vocab)
                            for i in sample["token_ids"]))
        self.assertTrue(all(tag in labels.TAG_TO_ID for tag in sample["tags"]))

    def test_batch_tensor_shapes_and_ignore_index(self):
        ids, targets = data_prep.make_batch(self.samples[:3])
        self.assertEqual(ids.dtype, torch.long)
        self.assertEqual(targets.dtype, torch.long)
        self.assertEqual(ids.shape, targets.shape)
        self.assertEqual(ids.shape[0], 3)
        # padded tag positions use the ignore index
        real = [len(s["token_ids"]) for s in self.samples[:3]]
        self.assertTrue((targets[:, max(real):] == data_prep.PAD_TARGET).all())
        self.assertTrue((targets[:, :min(real)] != data_prep.PAD_TARGET).all())

    def test_vocab_matches_phase4_artifact(self):
        phase4 = nlp_preprocess.load_vocab()
        with open(os.path.join(config.MODEL_DIR, "vocab.json"),
                  encoding="utf-8") as fh:
            saved = json.load(fh)["token_to_id"]
        self.assertEqual(phase4, saved)


class TestCaseSequenceToLstm(unittest.TestCase):
    """Requirement 17: sequence input -> LSTM (and gradients flow through it)."""

    @classmethod
    def setUpClass(cls):
        samples = data_prep.build_samples()
        cls.ids, cls.targets = data_prep.make_batch(samples[:4])
        cls.model = _load_model()

    def test_lstm_module_is_real(self):
        self.assertIsInstance(self.model.model.lstm, nn.LSTM)
        self.assertIsInstance(self.model.model, lstm_model.LstmExtractor)

    def test_forward_shape_and_finite(self):
        logits = self.model.model(self.ids)
        self.assertEqual(logits.shape,
                         (self.ids.shape[0], self.ids.shape[1],
                          len(labels.TAGS)))
        self.assertTrue(torch.isfinite(logits).all())

    def test_lstm_participates_in_gradients(self):
        model = _load_model().model
        model.train()
        logits = model(self.ids)
        loss = nn.CrossEntropyLoss(ignore_index=data_prep.PAD_TARGET)(
            logits.reshape(-1, model.num_tags), self.targets.reshape(-1))
        loss.backward()
        for name, parameter in model.lstm.named_parameters():
            self.assertIsNotNone(parameter.grad, f"no grad on {name}")
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        model.eval()


class TestCaseLstmToPrediction(unittest.TestCase):
    """Requirement 17: LSTM -> prediction."""

    @classmethod
    def setUpClass(cls):
        cls.extractor = _load_model()
        cls.documents = nlp_preprocess.load_documents()

    def test_prediction_tags_are_valid(self):
        sample = data_prep.build_samples()[0]
        predicted_ids = self.extractor.predict_tag_ids(sample["token_ids"])
        self.assertEqual(len(predicted_ids), len(sample["token_ids"]))
        self.assertTrue(all(index in labels.ID_TO_TAG
                            for index in predicted_ids))

    def test_trained_model_extracts_med1_entities(self):
        document = self.documents[0]
        result = self.extractor.extract_lines(document["parsed_lines"])
        by_type = {}
        for entity in result["entities"]:
            by_type.setdefault(entity["type"], []).append(entity["text"])

        self.assertEqual(by_type["BP"], ["127/82"])
        self.assertEqual(by_type["HR"], ["69 bpm"])
        self.assertEqual(by_type["GLU"], ["121 mg/dL"])
        self.assertEqual(by_type["CHOL"], ["151 mg/dL"])
        self.assertEqual(by_type["DX"], ["Acute Bronchitis"])
        self.assertEqual(by_type["MED"], ["Lisinopril"])
        self.assertEqual(by_type["DOSE"], ["10mg daily"])
        self.assertEqual(result["model"]["framework"], "pytorch")

    def test_entity_char_offsets_slice_original_line(self):
        document = self.documents[1]
        body_lines = nlp_preprocess.clinical_note_lines(document)
        result = self.extractor.extract_lines(document["parsed_lines"])
        self.assertTrue(result["entities"])
        for entity in result["entities"]:
            self.assertIsNotNone(entity["start_char"], entity)
            line = body_lines[entity["line_index"]]
            sliced = line[entity["start_char"]:entity["end_char"]]
            self.assertEqual(sliced, entity["text"])
            self.assertTrue(sliced)


class TestCaseSavedModelReload(unittest.TestCase):
    """Requirement 17: saved model -> reload -> prediction."""

    def test_reload_produces_identical_logits(self):
        sample = data_prep.build_samples()[0]
        first = _load_model().predict_logits(sample["token_ids"])
        second = _load_model().predict_logits(sample["token_ids"])
        self.assertTrue(torch.equal(first, second))

    def test_singleton_loads_once(self):
        inference.reset_extractor_for_tests()
        first = inference.get_extractor()
        second = inference.get_extractor()
        self.assertIs(first, second)

    def test_inference_never_retrains(self):
        weights = os.path.join(config.MODEL_DIR, "model_weights.pt")
        before = os.path.getmtime(weights)
        inference.reset_extractor_for_tests()
        extractor = inference.get_extractor()
        sample = data_prep.build_samples()[0]
        extractor.predict_tag_ids(sample["token_ids"])
        extractor.predict_tag_ids(sample["token_ids"])
        after = os.path.getmtime(weights)
        self.assertEqual(before, after, "weights file touched by inference!")

        # inference module contains no training machinery
        source = inspect.getsource(inference)
        self.assertNotIn("torch.optim", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("import train", source)


class TestCaseGatesMatchManualMath(unittest.TestCase):
    """Requirement 10: forget/input/output gates behave exactly as specified.

    Re-implements the LSTM equations from the raw weight matrices using
    the documented semantics:

        i_t = sigmoid(W_i x_t + U_i h_{t-1})     input gate (scales g_t)
        f_t = sigmoid(W_f x_t + U_f h_{t-1})     forget gate: elementwise
              multiplicative RETENTION control over c_{t-1} (scales what
              is carried forward; it does not delete text)
        g_t = tanh(W_g x_t + U_g h_{t-1})        candidate content
        o_t = sigmoid(W_o x_t + U_o h_{t-1})     output gate (scales
              how much of tanh(c_t) is exposed as h_t)
        c_t = f_t * c_{t-1} + i_t * g_t
        h_t = o_t * tanh(c_t)

    and asserts numerical equality with torch.nn.LSTM output.
    """

    def test_manual_gate_math_equals_nn_lstm(self):
        torch.manual_seed(7)
        input_size, hidden_size, batch, length = 8, 16, 2, 5
        lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        x = torch.randn(batch, length, input_size)
        output, _ = lstm(x)

        weight_ih = lstm.weight_ih_l0   # (4H, input)
        weight_hh = lstm.weight_hh_l0   # (4H, H)
        bias = lstm.bias_ih_l0 + lstm.bias_hh_l0
        h = torch.zeros(batch, hidden_size)
        c = torch.zeros(batch, hidden_size)
        H = hidden_size

        manual_outputs = []
        for step in range(length):
            gates = (x[:, step] @ weight_ih.T
                     + h @ weight_hh.T + bias)
            i_t = torch.sigmoid(gates[:, 0*H:1*H])      # input gate
            f_t = torch.sigmoid(gates[:, 1*H:2*H])      # forget gate (retention)
            g_t = torch.tanh(gates[:, 2*H:3*H])         # candidate
            o_t = torch.sigmoid(gates[:, 3*H:4*H])      # output gate
            c = f_t * c + i_t * g_t                     # cell state update
            h = o_t * torch.tanh(c)                     # exposed hidden state
            manual_outputs.append(h)

        manual = torch.stack(manual_outputs, dim=1)
        self.assertTrue(
            torch.allclose(manual, output, atol=1e-5),
            "manual gate equations diverge from torch.nn.LSTM")

    def test_forget_gate_scales_previous_cell_state(self):
        """f_t near 1 preserves c_{t-1}; f_t near 0 discards it."""
        torch.manual_seed(7)
        hidden = 4
        c_prev = torch.ones(1, hidden)

        retention = torch.full((1, hidden), 10.0)   # sigmoid -> ~1.0
        suppression = torch.full((1, hidden), -10.0)  # sigmoid -> ~0.0
        f_keep = torch.sigmoid(retention)
        f_drop = torch.sigmoid(suppression)

        kept = f_keep * c_prev
        dropped = f_drop * c_prev
        self.assertTrue(torch.allclose(kept, c_prev, atol=1e-3))
        self.assertLess(dropped.abs().max().item(), 1e-3)


class TestCaseLabelGeneration(unittest.TestCase):
    """Requirement 12: labels generated by documented rules, verified
    against the dataset's own reference labels for all 50 documents."""

    @classmethod
    def setUpClass(cls):
        cls.documents = nlp_preprocess.load_documents()
        cls.labeled = labels.label_documents(cls.documents)
        with open("extracted_medical_summary.py", encoding="utf-8") as fh:
            cls.reference = json.load(fh)

    @staticmethod
    def _collect(tokens, tags, entity_type):
        return " ".join(token for token, tag in zip(tokens, tags)
                        if tag in ("B-" + entity_type, "I-" + entity_type))

    def test_matches_reference_labels_for_all_50_documents(self):
        for document, labeled, reference in zip(
                self.documents, self.labeled, self.reference):
            key_points = reference["key_points"]
            tokens, tags = labeled["tokens"], labeled["tags"]
            doc_id = reference["document_id"]

            self.assertEqual(self._collect(tokens, tags, "BP"),
                             key_points["blood_pressure"], doc_id)
            self.assertEqual(self._collect(tokens, tags, "GLU").lower(),
                             key_points["sugar_level"].lower(), doc_id)
            self.assertEqual(self._collect(tokens, tags, "DX").lower(),
                             key_points["diagnosis"].rstrip(".").lower(),
                             doc_id)
            # one B-MED span per reference medication (name may be
            # multi-token, e.g. "albuterol inhaler" -> B-MED + I-MED)
            b_med_spans = sum(1 for tag in tags if tag == "B-MED")
            reference_names = [med.split(" - ")[0]
                               for med in key_points["medications"]]
            self.assertEqual(b_med_spans, len(reference_names), doc_id)

    def test_bio_sequences_have_no_orphan_i_tags(self):
        for labeled in self.labeled:
            previous = "O"
            for tag in labeled["tags"]:
                if tag.startswith("I-"):
                    self.assertIn(
                        previous, ("B-" + tag[2:], "I-" + tag[2:]),
                        f"orphan I- in {labeled['document_id']}")
                previous = tag

    def test_every_entity_type_appears_and_tokens_match_phase4(self):
        seen = set()
        for document, labeled in zip(self.documents, self.labeled):
            for tag in labeled["tags"]:
                if tag != "O":
                    seen.add(tag[2:])
            # identical token stream to Phase 4 preprocessing
            self.assertEqual(
                labeled["tokens"],
                nlp_preprocess.tokenize_document(document)["tokens"],
                document["document_id"])
        self.assertEqual(seen, set(labels.ENTITY_TYPES))


if __name__ == "__main__":
    unittest.main(verbosity=2)
