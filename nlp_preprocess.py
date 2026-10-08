"""NLP preprocessing pipeline for the synthetic medical document dataset.

Stage 1 of the ML pipeline: converts the raw clinical-note text of all 50
synthetic documents (MED-1 ... MED-50) into normalized text, tokens, a
vocabulary, and integer-encoded padded sequences - the exact tensor-ready
input an LSTM consumes.

Pure standard library on purpose. The deep-learning framework decision
(TensorFlow/Keras vs. PyTorch vs. from-scratch NumPy) is deliberately
deferred to the model phase, so nothing in this module may depend on it.

Pipeline:
    load_documents -> clinical_note_lines -> split_sentences
    -> normalize_text -> tokenize -> build_vocab -> encode/pad

Synthetic/demo data only; no real patient records are ever read here.
"""

import json
import os
import re
import unicodedata

import config

# ---------------------------------------------------------------------------
# Special tokens (fixed IDs - order must never change once artifacts exist)
# ---------------------------------------------------------------------------

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_ID = 0
UNK_ID = 1

DEFAULT_MIN_FREQ = 1

_START_MARKER = re.compile(r"---\s*START OF CLINICAL NOTES\s*---", re.IGNORECASE)
_END_MARKER = re.compile(r"---\s*END OF CLINICAL NOTES\s*---", re.IGNORECASE)

# Split only at a period/exclamation/question mark followed by whitespace and
# a capital letter - protected abbreviations (Dr., mg., ...) never split.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

_ABBREVIATIONS = ("dr", "mr", "mrs", "ms", "prof", "st", "vs", "etc",
                  "approx", "dept", "mg", "ml", "kg", "no")
_ABBR_RE = re.compile(
    r"\b(" + "|".join(_ABBREVIATIONS) + r")\.\s+", re.IGNORECASE)
_ABBR_SENTINEL = "\x00"

# Word tokenizer. Alternatives are tried in order at each position:
#   1. numbers / ratios / ranges / dates  (127/82, 70-99, 2025-11-24, 121, 200.5)
#   2. unit ratios                        (mg/dL -> "mg/dl")
#   3. words incl. hyphenated/possessive  (follow-up, patient's, covid-19)
#   4. any other non-whitespace character (punctuation, symbols)
_TOKEN_RE = re.compile(
    r"\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)*(?:/\d+(?:\.\d+)?)?%?"
    r"|[^\W\d_]+/[^\W\d_]+"
    r"|[^\W\d_]+(?:[-'](?:[^\W\d_]+|\d+))*"
    r"|[^\s]"
)

_SMART_PUNCTUATION = {
    "\u2018": "'",   # left single quote
    "\u2019": "'",   # right single quote / apostrophe
    "\u201c": '"',   # left double quote
    "\u201d": '"',   # right double quote
    "\u2013": "-",   # en dash
    "\u2014": "-",   # em dash
    "\u2212": "-",   # minus sign
    "\u00a0": " ",   # non-breaking space
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_documents(path=None):
    """Return the 50 synthetic documents as an ordered list of dicts.

    The dataset file contains a JSON array despite the .py extension.
    Raises ValueError when the file is missing or malformed.
    """
    path = path or config.MEDICAL_DOCUMENTS_FILE
    if not os.path.exists(path):
        raise ValueError("Medical dataset file is missing.")

    try:
        with open(path, "r", encoding="utf-8") as fh:
            documents = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Medical dataset could not be read: {exc}") from exc

    if not isinstance(documents, list) or not documents:
        raise ValueError("Medical dataset is empty or malformed.")
    return documents


# ---------------------------------------------------------------------------
# Line selection & sentence splitting
# ---------------------------------------------------------------------------

def clinical_note_lines(document, include_metadata=False):
    """Select which raw lines of a document enter the NLP pipeline.

    Default (include_metadata=False): only the lines between the
    "--- START/END OF CLINICAL NOTES ---" markers. Header fields
    (document ID, date, physician, patient name) are excluded because
    they are already handled by the structured lookup service and must
    not leak into model input. Documents without markers keep all lines.
    """
    lines = [line for line in document.get("parsed_lines", []) if line and line.strip()]

    if include_metadata:
        return lines

    start = None
    end = None
    for i, line in enumerate(lines):
        if start is None and _START_MARKER.search(line):
            start = i + 1
        elif start is not None and _END_MARKER.search(line):
            end = i
            break

    if start is None:
        return lines
    return lines[start:end] if end is not None else lines[start:]


def split_sentences(lines):
    """Split raw (original-case) lines into sentence strings.

    Abbreviations such as "Dr." or "mg." are temporarily sentinel-protected
    so they never count as sentence boundaries.
    """
    sentences = []
    for line in lines:
        protected = _ABBR_RE.sub(
            lambda match: match.group(1) + "." + _ABBR_SENTINEL, line.strip())
        for part in _SENTENCE_RE.split(protected):
            part = part.replace(_ABBR_SENTINEL, " ").strip()
            if part:
                sentences.append(part)
    return sentences


# ---------------------------------------------------------------------------
# Normalization & tokenization
# ---------------------------------------------------------------------------

def normalize_text(text):
    """NFKC-normalize, fold smart punctuation to ASCII, lowercase, squash space."""
    text = unicodedata.normalize("NFKC", text)
    for bad, good in _SMART_PUNCTUATION.items():
        text = text.replace(bad, good)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text):
    """Split normalized text into tokens (words, numbers, punctuation)."""
    return _TOKEN_RE.findall(normalize_text(text))


def tokenize_document(document, include_metadata=False):
    """Full per-document pass: lines -> sentences -> tokens.

    Returns {"document_id", "sentences": [[tokens], ...], "tokens": [...]}.
    """
    lines = clinical_note_lines(document, include_metadata=include_metadata)
    token_sentences = []
    for sentence in split_sentences(lines):
        tokens = tokenize(sentence)
        if tokens:
            token_sentences.append(tokens)

    return {
        "document_id": document.get("document_id", "N/A"),
        "sentences": token_sentences,
        "tokens": [token for sentence in token_sentences for token in sentence],
    }


# ---------------------------------------------------------------------------
# Vocabulary & encoding
# ---------------------------------------------------------------------------

def build_vocab(token_lists, min_freq=DEFAULT_MIN_FREQ):
    """Map token -> integer ID.

    IDs are deterministic: <PAD>=0, <UNK>=1, then tokens ordered by
    descending frequency (ties broken alphabetically). Tokens whose
    corpus frequency is below min_freq are excluded (encode as <UNK>).
    """
    counts = {}
    for tokens in token_lists:
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

    vocab = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    for token, count in ordered:
        if count >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_tokens(tokens, vocab):
    """Token strings -> integer IDs; unseen tokens become <UNK>."""
    return [vocab.get(token, UNK_ID) for token in tokens]


def decode_ids(ids, vocab):
    """Integer IDs -> token strings, skipping any padding."""
    id_to_token = {identifier: token for token, identifier in vocab.items()}
    return [id_to_token.get(identifier, UNK_TOKEN)
            for identifier in ids if identifier != PAD_ID]


def pad_sequence(ids, max_len, pad_id=PAD_ID):
    """Fit an ID sequence to exactly max_len: keep the head, pad the tail.

    Head-truncation is intentional - the opening lines of a clinical note
    carry the diagnosis context the model needs most.
    """
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [pad_id] * (max_len - len(ids))


# ---------------------------------------------------------------------------
# Whole-dataset pipeline
# ---------------------------------------------------------------------------

def preprocess_dataset(documents, min_freq=DEFAULT_MIN_FREQ,
                       include_metadata=False, max_len=None):
    """Tokenize every document, build the vocab, encode + pad all sequences.

    max_len=None keeps every sequence whole and pads the corpus to its
    longest document (lossless for this small dataset).
    Returns {"vocab", "sequences", "stats", "max_len"}.
    """
    tokenized = [tokenize_document(doc, include_metadata=include_metadata)
                 for doc in documents]
    token_lists = [item["tokens"] for item in tokenized]
    vocab = build_vocab(token_lists, min_freq=min_freq)

    lengths = [len(tokens) for tokens in token_lists]
    resolved_max = max_len or (max(lengths) if lengths else 0)

    sequences = [
        {
            "document_id": item["document_id"],
            "length": len(item["tokens"]),
            "token_ids": pad_sequence(
                encode_tokens(item["tokens"], vocab), resolved_max),
        }
        for item in tokenized
    ]

    ordered_lengths = sorted(lengths)
    stats = {
        "documents": len(documents),
        "total_tokens": sum(lengths),
        "unique_raw_tokens": len(set(token for tl in token_lists for token in tl)),
        "vocab_size": len(vocab),
        "min_freq": min_freq,
        "include_metadata": include_metadata,
        "sequence_length": {
            "min": ordered_lengths[0] if ordered_lengths else 0,
            "mean": round(sum(lengths) / len(lengths), 2) if lengths else 0,
            "p50": ordered_lengths[len(ordered_lengths) // 2] if ordered_lengths else 0,
            "max": ordered_lengths[-1] if ordered_lengths else 0,
        },
        "max_len": resolved_max,
        "truncated_sequences": sum(1 for item in sequences
                                   if item["length"] > resolved_max),
    }
    return {"vocab": vocab, "sequences": sequences, "stats": stats,
            "max_len": resolved_max}


# ---------------------------------------------------------------------------
# Artifact I/O
# ---------------------------------------------------------------------------

def save_artifacts(result, vocab_path=None, sequences_path=None, stats_path=None):
    """Persist vocab + sequences + stats as JSON under data/nlp/."""
    vocab_path = vocab_path or config.NLP_VOCAB_FILE
    sequences_path = sequences_path or config.NLP_SEQUENCES_FILE
    stats_path = stats_path or config.NLP_STATS_FILE

    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)

    id_to_token = [token for token, _ in
                   sorted(result["vocab"].items(), key=lambda item: item[1])]

    with open(vocab_path, "w", encoding="utf-8") as fh:
        json.dump({
            "token_to_id": result["vocab"],
            "id_to_token": id_to_token,
            "size": len(result["vocab"]),
            "min_freq": result["stats"]["min_freq"],
            "max_len": result["max_len"],
        }, fh, ensure_ascii=False, indent=2)

    with open(sequences_path, "w", encoding="utf-8") as fh:
        json.dump(result["sequences"], fh, ensure_ascii=False)

    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(result["stats"], fh, ensure_ascii=False, indent=2)


def load_vocab(vocab_path=None):
    """Reload a saved vocabulary (token -> id)."""
    vocab_path = vocab_path or config.NLP_VOCAB_FILE
    with open(vocab_path, "r", encoding="utf-8") as fh:
        return json.load(fh)["token_to_id"]


def load_sequences(sequences_path=None):
    """Reload saved encoded sequences."""
    sequences_path = sequences_path or config.NLP_SEQUENCES_FILE
    with open(sequences_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# CLI: build artifacts and print the phase report
# ---------------------------------------------------------------------------

def main():
    documents = load_documents()
    result = preprocess_dataset(documents)
    save_artifacts(result)

    stats = result["stats"]
    seq_len = stats["sequence_length"]

    print("=== Phase 4: NLP Preprocessing Report ===")
    print(f"documents processed      : {stats['documents']}")
    print(f"total tokens             : {stats['total_tokens']}")
    print(f"raw unique tokens        : {stats['unique_raw_tokens']}")
    print(f"vocab size (min_freq={stats['min_freq']})   : {stats['vocab_size']}")
    print(f"sequence length min/avg  : {seq_len['min']} / {seq_len['mean']}")
    print(f"sequence length p50/max  : {seq_len['p50']} / {seq_len['max']}")
    print(f"padded to max_len        : {result['max_len']}")
    print(f"truncated sequences      : {stats['truncated_sequences']}")
    print()
    print("artifacts written:")
    print(f"  {config.NLP_VOCAB_FILE}")
    print(f"  {config.NLP_SEQUENCES_FILE}")
    print(f"  {config.NLP_STATS_FILE}")
    print()

    sample = tokenize_document(documents[0])
    print(f"sample tokenization ({sample['document_id']}, first 25 tokens):")
    print("  " + " | ".join(sample["tokens"][:25]))
    print()
    print("vocabulary (first 20 by rank):")
    ranked = sorted(result["vocab"].items(), key=lambda item: item[1])[2:22]
    print("  " + ", ".join(f"{token}={identifier}" for token, identifier in ranked))


if __name__ == "__main__":
    main()
