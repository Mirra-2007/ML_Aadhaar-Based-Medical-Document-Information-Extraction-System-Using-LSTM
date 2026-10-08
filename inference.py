"""Inference for Phase 5: load the saved LSTM once, reuse it forever.

Design rules (Phase 5 requirements):
* NEVER trains or retrains anything - this module contains no optimizer,
  no loss, and no weight updates. It only loads and runs forward passes.
* The trained model + vocabulary + tag map are loaded ONCE into a
  process-wide singleton (`get_extractor`) and reused for every call,
  so a patient lookup or extraction never pays the load cost twice and
  never touches the training artifacts' modification time.
* All required artifacts live together under model/ (self-contained:
  weights + config + vocab copy + tag map).

Usage:
    from inference import get_extractor
    extractor = get_extractor()                 # loads on first call only
    result = extractor.extract_text(note_text)  # -> entities
"""

import json
import os

import torch

import config
import labels
import lstm_model
import nlp_preprocess


def _read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _body_lines(lines):
    """Same body-only selection used in training (headers excluded).

    Keeps inference input identical to what the model was trained on:
    tokens outside the clinical-note body (DOCUMENT ID, DATE, ...) are
    out-of-vocabulary and were never seen during training.
    """
    return nlp_preprocess.clinical_note_lines(
        {"parsed_lines": list(lines)}, include_metadata=False)


def _line_token_offsets(raw_line, tokens, normalized):
    """Exact char offsets of tokens within the ORIGINAL line.

    Mirrors labels.py: tokens partition the normalized line, so sequential
    substring search from a monotonic cursor yields each token's true
    position (substring collisions such as '89' inside '136/89' are
    disambiguated by the cursor). Normalization lowercases and strips;
    normalization is length-preserving for this dataset, so normalized
    offsets map onto the original (unstripped) line after offsetting by
    the removed leading whitespace. Returns None when normalization
    altered the text non-trivially (callers fall back to normalized text).
    """
    stripped = raw_line.strip()
    if stripped.lower() != normalized:
        return None
    prefix = len(raw_line) - len(raw_line.lstrip())
    offsets = []
    cursor = 0
    for token in tokens:
        position = normalized.find(token, cursor)
        if position == -1:
            return None
        offsets.append(prefix + position)
        cursor = position + len(token)
    return offsets


class Extractor:
    """Saved-model wrapper: preprocess -> LSTM forward -> BIO decode."""

    def __init__(self, model, vocab, tag_to_id, model_config):
        self.model = model
        self.vocab = vocab
        self.tag_to_id = tag_to_id
        self.id_to_tag = {int(index): tag for tag, index in tag_to_id.items()}
        self.model_config = model_config
        self.model.eval()

    # -- loading ----------------------------------------------------------

    @classmethod
    def load(cls, model_dir=None):
        model_dir = model_dir or config.MODEL_DIR
        weights_path = os.path.join(model_dir, "model_weights.pt")
        config_path = os.path.join(model_dir, "model_config.json")
        vocab_path = os.path.join(model_dir, "vocab.json")
        tags_path = os.path.join(model_dir, "tags.json")

        for path in (weights_path, config_path, vocab_path, tags_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Model artifact missing: {path}. "
                    "Run `py train.py` to train and save the model first.")

        model_config = _read_json(config_path)
        vocab = _read_json(vocab_path)["token_to_id"]
        tag_to_id = _read_json(tags_path)["tag_to_id"]

        model = lstm_model.LstmExtractor(
            vocab_size=model_config["vocab_size"],
            embed_dim=model_config["embed_dim"],
            hidden_dim=model_config["hidden_dim"],
            num_tags=model_config["num_tags"],
            pad_id=model_config["pad_id"],
            num_layers=model_config["num_layers"],
        )
        state_dict = torch.load(weights_path, map_location="cpu",
                                weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        return cls(model, vocab, tag_to_id, model_config)

    # -- prediction -------------------------------------------------------

    @torch.no_grad()
    def predict_tag_ids(self, token_ids):
        """[token ids] -> [predicted tag ids] (single unpadded sequence)."""
        ids = torch.tensor([token_ids], dtype=torch.long)
        logits = self.model(ids)
        return logits.argmax(dim=-1)[0].tolist()

    @torch.no_grad()
    def predict_logits(self, token_ids):
        """[token ids] -> logits tensor (1, length, num_tags)."""
        ids = torch.tensor([token_ids], dtype=torch.long)
        return self.model(ids)

    def extract_lines(self, lines):
        """Extract entities from raw clinical-note lines.

        Input is filtered with the same body-only selection used in
        training (header lines are excluded). The whole note body is
        then tokenized and pushed through the LSTM AS ONE SEQUENCE -
        identical to the training input format - so cross-line context
        flows through the cell state; predicted tags are afterwards
        attributed back to their source lines.

        Line indices in the output are relative to the clinical-note
        body (the model input). Each entity is:
            {"type", "text", "line_index", "token_start", "token_end",
             "start_char", "end_char"}   (char offsets in that body line)
        """
        body_lines = _body_lines(lines)

        line_data = []          # (raw_line, tokens)
        token_ids = []
        for raw_line in body_lines:
            normalized = nlp_preprocess.normalize_text(raw_line)
            tokens = nlp_preprocess.tokenize(normalized) if normalized else []
            line_data.append((raw_line, tokens))
            token_ids.extend(nlp_preprocess.encode_tokens(tokens, self.vocab))

        if not token_ids:
            return {"entities": [], "model": self._model_info()}

        predicted_ids = self.predict_tag_ids(token_ids)

        entities = []
        cursor = 0
        for line_index, (raw_line, tokens) in enumerate(line_data):
            line_predicted = predicted_ids[cursor:cursor + len(tokens)]
            cursor += len(tokens)
            if not tokens:
                continue
            predicted_tags = [self.id_to_tag[index]
                              for index in line_predicted]

            normalized = nlp_preprocess.normalize_text(raw_line)
            offsets = _line_token_offsets(raw_line, tokens, normalized)

            for token_start, token_end, entity_type in _decode_bio(predicted_tags):
                if offsets is not None:
                    first, last = token_start, token_end - 1
                    start_char = offsets[first]
                    end_char = offsets[last] + len(tokens[last])
                    original_text = raw_line[start_char:end_char]
                else:  # normalization altered the source text
                    start_char = end_char = None
                    original_text = " ".join(tokens[token_start:token_end])
                entities.append({
                    "type": entity_type,
                    "text": original_text,
                    "line_index": line_index,
                    "token_start": token_start,
                    "token_end": token_end,
                    "start_char": start_char,
                    "end_char": end_char,
                })

        return {"entities": entities, "model": self._model_info()}

    def _model_info(self):
        return {
            "framework": self.model_config.get("framework", "pytorch"),
            "epochs_trained": (self.model_config.get("hyperparameters") or {})
                .get("epochs_trained"),
            "label_rules_version": labels.LABEL_RULES_VERSION,
        }

    def extract_text(self, text):
        """Convenience wrapper: split a full note into lines and extract."""
        return self.extract_lines(text.splitlines())


def _decode_bio(tag_sequence):
    """BIO tag list -> [(token_start, token_end_exclusive, type), ...]."""
    spans = []
    entity_type = None
    start = None
    for index, tag in enumerate(tag_sequence):
        if tag.startswith("B-"):
            if entity_type is not None:
                spans.append((start, index, entity_type))
            entity_type = tag[2:]
            start = index
        elif tag.startswith("I-") and entity_type == tag[2:]:
            continue
        else:
            if entity_type is not None:
                spans.append((start, index, entity_type))
            entity_type = None
            start = None
    if entity_type is not None:
        spans.append((start, len(tag_sequence), entity_type))
    return spans


# ---------------------------------------------------------------------------
# Process-wide singleton: load once, reuse for every call
# ---------------------------------------------------------------------------

_EXTRACTOR = None


def get_extractor():
    """Return the shared Extractor, loading it exactly once per process."""
    global _EXTRACTOR
    if _EXTRACTOR is None:
        _EXTRACTOR = Extractor.load()
    return _EXTRACTOR


def reset_extractor_for_tests():
    """Drop the singleton (tests only; never needed in application flow)."""
    global _EXTRACTOR
    _EXTRACTOR = None
