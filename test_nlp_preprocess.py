"""Unit and integration tests for the Phase 4 NLP preprocessing pipeline.

Run from the project root:

    py -m unittest test_nlp_preprocess -v
"""

import os
import tempfile
import unittest

import nlp_preprocess as nlp


class NormalizeTests(unittest.TestCase):
    def test_lowercases_and_collapses_whitespace(self):
        self.assertEqual(nlp.normalize_text("  BP:\t127/82 \n\n HR:  69  "),
                         "bp: 127/82 hr: 69")

    def test_smart_punctuation_folded_to_ascii(self):
        self.assertEqual(nlp.normalize_text("patient\u2019s\u201cvoice\u201d"),
                         "patient's\"voice\"")

    def test_dashes_and_nonbreaking_space(self):
        self.assertEqual(nlp.normalize_text("a\u2014b\u00a0c\u2212d"), "a-b c-d")


class TokenizerTests(unittest.TestCase):
    def tokens(self, text):
        return nlp.tokenize(text)

    def test_vital_signs_line(self):
        self.assertEqual(self.tokens("BP: 127/82, HR: 69 bpm."),
                         ["bp", ":", "127/82", ",", "hr", ":", "69", "bpm", "."])

    def test_number_with_unit_splits_number_from_unit(self):
        self.assertEqual(self.tokens("Metformin - 500mg twice daily"),
                         ["metformin", "-", "500", "mg", "twice", "daily"])

    def test_unit_ratio_stays_single_token(self):
        self.assertEqual(self.tokens("121 mg/dL"), ["121", "mg/dl"])

    def test_reference_range_stays_single_token(self):
        self.assertEqual(self.tokens("(Normal: 70-99 mg/dL)"),
                         ["(", "normal", ":", "70-99", "mg/dl", ")"])

    def test_date_stays_single_token(self):
        self.assertEqual(self.tokens("2025-11-24"), ["2025-11-24"])

    def test_abbreviation_splits_period_off(self):
        self.assertEqual(self.tokens("Dr. XXXXX, MD"),
                         ["dr", ".", "xxxxx", ",", "md"])

    def test_hyphenated_word_stays_single_token(self):
        self.assertEqual(self.tokens("routine follow-up evaluation"),
                         ["routine", "follow-up", "evaluation"])

    def test_less_than_sign_kept(self):
        self.assertEqual(self.tokens("Cholesterol < 200 mg/dL"),
                         ["cholesterol", "<", "200", "mg/dl"])

    def test_pure_numeric_document_id_is_one_token(self):
        self.assertEqual(self.tokens("#MED-1"), ["#", "med-1"])


class SentenceSplitTests(unittest.TestCase):
    def test_splits_only_at_period_followed_by_capital(self):
        sentences = nlp.split_sentences(
            ["Primary Assessment / Diagnosis: Type 2 Diabetes Mellitus. "
             "Patient advised to maintain diet."])
        self.assertEqual(len(sentences), 2)
        self.assertTrue(sentences[0].endswith("Mellitus."))
        self.assertTrue(sentences[1].startswith("Patient advised"))

    def test_abbreviations_do_not_split(self):
        sentences = nlp.split_sentences(["Reviewed by Dr. Smith, MD today."])
        self.assertEqual(len(sentences), 1)

    def test_blank_lines_ignored(self):
        self.assertEqual(nlp.split_sentences(["", "   ", "Single line."]),
                         ["Single line."])


class LineSelectionTests(unittest.TestCase):
    DOC = {
        "document_id": "MED-1",
        "parsed_lines": [
            "DOCUMENT ID: #MED-1",
            "DATE OF ENCOUNTER: 2025-11-24",
            "--- START OF CLINICAL NOTES ---",
            "Chief Complaint: Patient reports fatigue.",
            "",
            " --- END OF CLINICAL NOTES ---",
        ],
    }

    def test_default_excludes_header_and_markers(self):
        lines = nlp.clinical_note_lines(self.DOC)
        self.assertEqual(lines, ["Chief Complaint: Patient reports fatigue."])

    def test_include_metadata_keeps_everything(self):
        lines = nlp.clinical_note_lines(self.DOC, include_metadata=True)
        self.assertEqual(len(lines), 5)

    def test_document_without_markers_keeps_all_lines(self):
        doc = {"parsed_lines": ["No markers here.", "Second line."]}
        self.assertEqual(nlp.clinical_note_lines(doc),
                         ["No markers here.", "Second line."])


class VocabularyEncodingTests(unittest.TestCase):
    TOKENS = ["bp", "127/82", "bp", "hr", "hr", "hr"]

    def test_special_ids_are_fixed(self):
        vocab = nlp.build_vocab([self.TOKENS])
        self.assertEqual(vocab[nlp.PAD_TOKEN], 0)
        self.assertEqual(vocab[nlp.UNK_TOKEN], 1)

    def test_frequency_order_then_alphabetical(self):
        vocab = nlp.build_vocab([self.TOKENS])
        # counts: hr=3, bp=2, 127/82=1  ->  hr=2, bp=3, 127/82=4
        self.assertEqual(vocab["hr"], 2)
        self.assertEqual(vocab["bp"], 3)
        self.assertEqual(vocab["127/82"], 4)
        # deterministic: same input twice -> identical vocab
        self.assertEqual(vocab, nlp.build_vocab([self.TOKENS]))

    def test_min_freq_filters_rare_tokens_to_unk(self):
        vocab = nlp.build_vocab([self.TOKENS], min_freq=2)
        self.assertNotIn("127/82", vocab)
        self.assertEqual(nlp.encode_tokens(["127/82"], vocab), [nlp.UNK_ID])

    def test_encode_decode_roundtrip(self):
        vocab = nlp.build_vocab([["bp", "hr", "127/82"]])
        ids = nlp.encode_tokens(["bp", "hr", "127/82"], vocab)
        self.assertEqual(nlp.decode_ids(ids, vocab), ["bp", "hr", "127/82"])

    def test_unknown_token_encodes_to_unk(self):
        vocab = nlp.build_vocab([["bp"]])
        self.assertEqual(nlp.encode_tokens(["zzz"], vocab), [nlp.UNK_ID])

    def test_pad_pads_tail(self):
        self.assertEqual(nlp.pad_sequence([2, 3], 5), [2, 3, 0, 0, 0])

    def test_pad_truncates_head_keeps_head(self):
        self.assertEqual(nlp.pad_sequence([2, 3, 4, 5, 6], 3), [2, 3, 4])


class DatasetIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents = nlp.load_documents()
        cls.result = nlp.preprocess_dataset(cls.documents)

    def test_all_50_documents_processed(self):
        self.assertEqual(len(self.documents), 50)
        self.assertEqual(len(self.result["sequences"]), 50)

    def test_no_empty_sequences(self):
        for seq in self.result["sequences"]:
            self.assertGreater(seq["length"], 0, seq["document_id"])

    def test_all_sequences_padded_to_same_length(self):
        for seq in self.result["sequences"]:
            self.assertEqual(len(seq["token_ids"]), self.result["max_len"])

    def test_no_truncation_when_max_len_unset(self):
        self.assertEqual(self.result["stats"]["truncated_sequences"], 0)

    def test_header_fields_do_not_leak_into_tokens(self):
        joined = " ".join(nlp.tokenize_document(self.documents[0])["tokens"])
        self.assertNotIn("document", joined)
        self.assertNotIn("encounter", joined)
        self.assertNotIn("2025-11-24", joined)
        self.assertIn("diagnosis", joined)
        self.assertIn("bronchitis", joined)

    def test_medical_values_survive_tokenization(self):
        joined = " ".join(nlp.tokenize_document(self.documents[0])["tokens"])
        self.assertIn("127/82", joined)
        self.assertIn("mg/dl", joined)

    def test_vocab_is_deterministic(self):
        second = nlp.preprocess_dataset(nlp.load_documents())
        self.assertEqual(self.result["vocab"], second["vocab"])
        self.assertEqual(self.result["sequences"], second["sequences"])

    def test_min_freq_2_still_covers_diagnosis_terms(self):
        result = nlp.preprocess_dataset(self.documents, min_freq=2)
        for term in ("diagnosis", "patient", "bronchitis"):
            self.assertIn(term, result["vocab"], term)


class ArtifactIOTests(unittest.TestCase):
    def test_save_and_reload_roundtrip(self):
        documents = nlp.load_documents()
        result = nlp.preprocess_dataset(documents[:3])
        with tempfile.TemporaryDirectory() as tmp:
            vocab_path = os.path.join(tmp, "vocab.json")
            seq_path = os.path.join(tmp, "sequences.json")
            stats_path = os.path.join(tmp, "stats.json")
            nlp.save_artifacts(result, vocab_path, seq_path, stats_path)

            self.assertEqual(nlp.load_vocab(vocab_path), result["vocab"])
            self.assertEqual(nlp.load_sequences(seq_path), result["sequences"])
            self.assertTrue(os.path.getsize(stats_path) > 0)

    def test_missing_dataset_file_raises(self):
        with self.assertRaises(ValueError):
            nlp.load_documents(path=os.path.join("no", "such", "file.json"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
