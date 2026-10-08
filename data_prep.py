"""Data preparation for Phase 5 (labeled samples -> model-ready tensors).

Separation of concerns: this module only loads, labels, encodes and
splits. It never builds or trains a model.

Pipeline:  dataset (Phase 4) -> labels.py -> Phase 4 vocab encoding
           -> deterministic train/val/test split -> padded batches.

SPLIT RULE (deterministic, no shuffle, no randomness)
=====================================================
Documents are ordered by numeric id (MED-1 ... MED-50), then split by
original index i (0-based):

    i % 5 == 4  -> test   (10 documents: MED-5, 10, 15, ... 50)
    i % 5 == 3  -> val    (10 documents: MED-4, 9, 14, ... 49)
    else        -> train  (30 documents)

Selection/model-choosing uses val only; the test split is evaluated
once at the end. Phase 5's final model is retrained on train+val
(40 documents) for the selected epoch count - see train.py.
"""

import nlp_preprocess
import labels

PAD_TARGET = -100  # CrossEntropyLoss ignore_index for padded tag positions


def build_samples():
    """Encode every document: ids from the Phase 4 vocab, tags from labels.py.

    Returns a list of dicts:
        {"document_id", "tokens", "token_ids", "tags"} with
        len(tokens) == len(token_ids) == len(tags).
    """
    documents = nlp_preprocess.load_documents()
    vocab = nlp_preprocess.load_vocab()
    labeled = labels.label_documents(documents)

    samples = []
    for item in labeled:
        token_ids = nlp_preprocess.encode_tokens(item["tokens"], vocab)
        samples.append({
            "document_id": item["document_id"],
            "tokens": item["tokens"],
            "token_ids": token_ids,
            "tags": item["tags"],
        })
    return samples


def _numeric_id(sample):
    return int(sample["document_id"].split("-")[1])


def split_samples(samples=None):
    """Apply the deterministic 30/10/10 split rule documented above."""
    if samples is None:
        samples = build_samples()
    ordered = sorted(samples, key=_numeric_id)

    train, val, test = [], [], []
    for index, sample in enumerate(ordered):
        remainder = index % 5
        if remainder == 4:
            test.append(sample)
        elif remainder == 3:
            val.append(sample)
        else:
            train.append(sample)
    return train, val, test


def make_batch(sample_list, max_len=None):
    """Pad one batch (list of samples) -> (ids, targets) LongTensors.

    Rows are padded to the batch's longest sequence (or max_len), token
    ids with PAD_ID=0, tag targets with PAD_TARGET=-100 so padded
    positions contribute nothing to the loss or metrics.
    """
    import torch  # local import: keeps module importable without torch for labels-only use

    lengths = [len(sample["token_ids"]) for sample in sample_list]
    length = max(lengths) if max_len is None else max_len

    ids = torch.zeros((len(sample_list), length), dtype=torch.long)
    targets = torch.full((len(sample_list), length), PAD_TARGET, dtype=torch.long)

    for row, sample in enumerate(sample_list):
        seq_len = len(sample["token_ids"])
        ids[row, :seq_len] = torch.tensor(sample["token_ids"], dtype=torch.long)
        tag_ids = [labels.TAG_TO_ID[tag] for tag in sample["tags"]]
        targets[row, :seq_len] = torch.tensor(tag_ids, dtype=torch.long)
    return ids, targets
