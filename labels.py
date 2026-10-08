r"""Label generation for Phase 5 (LSTM information extraction).

TASK DESIGN
===========
Token-level information extraction (BIO sequence labeling): given the
clinical-note body of a synthetic document, tag every token as belonging
to a clinically relevant field or not. This is the "distinguish/retrieve
clinically relevant information from the clinical-note text" objective
in sequence-labeling form - the standard formulation for medical
information extraction.

HOW LABELS ARE GENERATED (deterministic rule-based / weak supervision)
======================================================================
Labels are derived from the existing 50-document dataset only - no
hand-annotation, no external data, no fabricated fields.

For every document, the clinical-note body is selected with Phase 4's
`nlp_preprocess.clinical_note_lines(include_metadata=False)` (header
lines such as DOCUMENT ID / DATE / PATIENT NAME are excluded, exactly
like the model input).

Each line is normalized with `nlp_preprocess.normalize_text`, then up to
one value span per rule is located by regex on the normalized line:

  RULE 1  line matches `BP:\s*([\d/]+)`
          -> span tagged BP          (e.g. "127/82")

  RULE 2  line matches `HR:\s*(\d+\s*bpm)`
          -> span tagged HR          (e.g. "69 bpm")

  RULE 3  line matches `Blood Glucose:\s*([\d]+\s*mg/dL)`
          -> first match only, tagged GLU   (e.g. "121 mg/dL");
             the "(Normal: ...)" reference range is NOT labeled

  RULE 4  line matches `Cholesterol Total:\s*([\d]+\s*mg/dL)`
          -> first match only, tagged CHOL  (e.g. "151 mg/dL");
             the reference range is NOT labeled

  RULE 5  line matches `Diagnosis:\s*(.+)`
          -> value tagged DX, trailing period excluded
             (e.g. "acute bronchitis")

  RULE 6  line matches `\*\s*Rx:\s*(.+)$`
          -> value split at the first " - ": name part tagged MED,
             dosage part tagged DOSE
             (e.g. "lisinopril" / "10mg daily")

  any line/region matching no rule -> O

Token-to-span alignment: the line's tokens (Phase 4 tokenizer) are
located inside the normalized line by sequential substring search - an
exact method for this tokenizer because every non-whitespace character
is consumed by a token, so inter-token gaps contain only whitespace.
A token whose start lies at or inside a span start receives B-<TYPE>;
tokens fully inside the span afterwards receive I-<TYPE>. Spans never
overlap by construction (each rule targets disjoint line regions).

The same token stream is what the model sees, so labels are always
aligned 1:1 with model inputs.

BIO tag set (14 tags; O has id 0):
    O, B-BP, B-HR, I-HR, B-GLU, I-GLU, B-CHOL, I-CHOL,
    B-DX, I-DX, B-MED, I-MED, B-DOSE, I-DOSE

Consistency with the dataset's own label file
(`extracted_medical_summary.py` key_points) is asserted by
`test_lstm.TestCaseLabelGeneration` for all 50 documents.
"""

import re

import nlp_preprocess

# ---------------------------------------------------------------------------
# Tag vocabulary (fixed: order defines IDs and must not change)
# ---------------------------------------------------------------------------

TAGS = [
    "O",
    "B-BP", "B-HR", "I-HR",
    "B-GLU", "I-GLU",
    "B-CHOL", "I-CHOL",
    "B-DX", "I-DX",
    "B-MED", "I-MED",
    "B-DOSE", "I-DOSE",
]

TAG_TO_ID = {tag: index for index, tag in enumerate(TAGS)}
ID_TO_TAG = {index: tag for tag, index in TAG_TO_ID.items()}

ENTITY_TYPES = ["BP", "HR", "GLU", "CHOL", "DX", "MED", "DOSE"]

LABEL_RULES_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Value-span rules (regexes run on the NORMALIZED (lowercased) line, so
# they are compiled case-insensitively)
# ---------------------------------------------------------------------------

_RULE_BP = re.compile(r"BP:\s*([\d/]+)", re.IGNORECASE)
_RULE_HR = re.compile(r"HR:\s*(\d+\s*bpm)", re.IGNORECASE)
_RULE_GLU = re.compile(r"Blood Glucose:\s*([\d]+\s*mg/dL)", re.IGNORECASE)
_RULE_CHOL = re.compile(r"Cholesterol Total:\s*([\d]+\s*mg/dL)", re.IGNORECASE)
_RULE_DX = re.compile(r"Diagnosis:\s*(.+)", re.IGNORECASE)
_RULE_RX = re.compile(r"\*\s*Rx:\s*(.+)$", re.IGNORECASE)


def _span(match, group, entity, strip_trailing_period=False):
    start, end = match.start(group), match.end(group)
    if strip_trailing_period and end > start and match.group(group).endswith("."):
        end -= 1
    return (start, end, entity) if end > start else None


def _line_value_spans(normalized_line):
    """Apply rules 1-6 to one normalized line -> list of (start, end, type)."""
    spans = []

    match = _RULE_BP.search(normalized_line)
    if match:
        spans.append(_span(match, 1, "BP"))

    match = _RULE_HR.search(normalized_line)
    if match:
        spans.append(_span(match, 1, "HR"))

    match = _RULE_GLU.search(normalized_line)
    if match:
        spans.append(_span(match, 1, "GLU"))

    match = _RULE_CHOL.search(normalized_line)
    if match:
        spans.append(_span(match, 1, "CHOL"))

    match = _RULE_DX.search(normalized_line)
    if match:
        span = _span(match, 1, "DX", strip_trailing_period=True)
        if span:
            spans.append(span)

    match = _RULE_RX.search(normalized_line)
    if match:
        value_start = match.start(1)
        value = match.group(1)
        separator = value.find(" - ")
        if separator != -1:
            # name part -> MED, dosage part -> DOSE
            spans.append((value_start, value_start + separator, "MED"))
            dose_start = value_start + separator + len(" - ")
            spans.append((dose_start, value_start + len(value), "DOSE"))
        else:
            spans.append((value_start, value_start + len(value), "MED"))

    return [span for span in spans if span is not None]


def _token_offsets(normalized_line, tokens):
    """Exact start offsets of each token inside the normalized line."""
    offsets = []
    cursor = 0
    for token in tokens:
        position = normalized_line.find(token, cursor)
        if position == -1:  # unreachable for this tokenizer; defensive
            position = cursor
        offsets.append(position)
        cursor = position + len(token)
    return offsets


# ---------------------------------------------------------------------------
# Document labeling
# ---------------------------------------------------------------------------

def label_document(document):
    """Generate (tokens, tags) for one dataset document.

    Returns {"document_id", "tokens": [str], "tags": [str]} where
    len(tokens) == len(tags), driven by the same Phase 4 line selection
    and tokenizer that produce the model input.
    """
    all_tokens = []
    all_tags = []

    lines = nlp_preprocess.clinical_note_lines(document, include_metadata=False)
    for line in lines:
        normalized = nlp_preprocess.normalize_text(line)
        if not normalized:
            continue

        spans = _line_value_spans(normalized)
        tokens = nlp_preprocess.tokenize(normalized)
        if not tokens:
            continue
        offsets = _token_offsets(normalized, tokens)

        tags = []
        for index, token in enumerate(tokens):
            start = offsets[index]
            end = start + len(token)
            tag = "O"
            for span_start, span_end, entity in spans:
                if end > span_start and start < span_end:
                    # token containing/after the span start: B if it starts
                    # at or before the span start, otherwise continuation I
                    tag = "B-" + entity if start <= span_start else "I-" + entity
                    break
            tags.append(tag)

        all_tokens.extend(tokens)
        all_tags.extend(tags)

    return {
        "document_id": document.get("document_id", "N/A"),
        "tokens": all_tokens,
        "tags": all_tags,
    }


def label_documents(documents):
    """Label every document; raises if any (tokens, tags) misalign."""
    labeled = []
    for document in documents:
        item = label_document(document)
        if len(item["tokens"]) != len(item["tags"]):
            raise ValueError(
                f"Label misalignment in {item['document_id']}: "
                f"{len(item['tokens'])} tokens vs {len(item['tags'])} tags")
        labeled.append(item)
    return labeled
