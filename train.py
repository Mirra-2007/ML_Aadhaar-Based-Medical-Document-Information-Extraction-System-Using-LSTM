"""Training, evaluation and model saving for Phase 5 (PyTorch LSTM).

Separation of concerns: this file owns training + evaluation + saving.
Run it once:  `py train.py`   (never runs during inference/lookup).

Two-phase protocol (test split untouched until the very end):
  Phase A  train on 30 docs, choose epoch count by val loss on 10 docs
           (early stopping with patience).
  Phase B  retrain a fresh, identically-seeded model on train+val (40
           docs) for the chosen epoch count; evaluate once on the 10
           test docs; save artifacts under model/.

Metrics: token accuracy, per-tag precision/recall/F1 + macro-F1
(excluding O), and entity-level exact-span precision/recall/F1.
"""

import json
import os
import platform
import random
import sys
import time

import torch
import torch.nn as nn

import config
import data_prep
import labels
import nlp_preprocess
from lstm_model import LstmExtractor

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

SEED = 42
EMBED_DIM = 32
HIDDEN_DIM = 64
NUM_LAYERS = 1
LEARNING_RATE = 1e-3
MAX_EPOCHS = 150
PATIENCE = 15
BATCH_SIZE = 8

MODEL_WEIGHTS = "model_weights.pt"
MODEL_CONFIG = "model_config.json"
MODEL_VOCAB = "vocab.json"
MODEL_TAGS = "tags.json"
MODEL_METRICS = "metrics.json"
MODEL_SPLIT = "split.json"

LIMITATIONS = [
    "Corpus is 50 synthetic documents from a single template family; "
    "test documents share near-identical structure with train documents, "
    "so held-out scores are optimistic and say nothing about real "
    "clinical language.",
    "Labels are weak supervision (deterministic rules over the same "
    "templates), so the model is scored on learning extraction patterns, "
    "not on clinical understanding.",
    "Vital/lab values and diagnoses were randomized independently when "
    "the dataset was generated; there is no causal signal beyond surface "
    "position/context patterns.",
    "Single seeded run, no cross-validation; vocabulary is tiny (Phase 4 "
    "vocab of ~229 tokens) and English-only.",
    "Entity types are limited to the seven dataset fields (BP, HR, GLU, "
    "CHOL, DX, MED, DOSE).",
]


# ---------------------------------------------------------------------------
# Seeding & batches
# ---------------------------------------------------------------------------

def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)


def _batches(samples, batch_size=BATCH_SIZE):
    for start in range(0, len(samples), batch_size):
        yield samples[start:start + batch_size]


def _build_model(vocab_size, num_tags):
    return LstmExtractor(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_tags=num_tags,
        pad_id=0,
        num_layers=NUM_LAYERS,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, optimizer, criterion, samples):
    model.train()
    loss_sum = 0.0
    correct = 0
    tokens = 0
    for batch in _batches(samples):
        ids, targets = data_prep.make_batch(batch)
        optimizer.zero_grad()
        logits = model(ids)
        loss = criterion(logits.reshape(-1, model.num_tags),
                         targets.reshape(-1))
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            mask = targets != data_prep.PAD_TARGET
            count = mask.sum().item()
            predicted = logits.argmax(dim=-1)
            correct += ((predicted == targets) & mask).sum().item()
            tokens += count
            loss_sum += loss.item() * count
    return loss_sum / max(tokens, 1), correct / max(tokens, 1)


def _loss_only(model, criterion, samples):
    model.eval()
    loss_sum, tokens = 0.0, 0
    with torch.no_grad():
        for sample in samples:
            ids, targets = data_prep.make_batch([sample])
            logits = model(ids)
            loss = criterion(logits.reshape(-1, model.num_tags),
                             targets.reshape(-1))
            count = targets.ne(data_prep.PAD_TARGET).sum().item()
            loss_sum += loss.item() * count
            tokens += count
    return loss_sum / max(tokens, 1)


def select_epoch_count(train_samples, val_samples, vocab_size, num_tags):
    """Phase A: train on train split, pick best epoch by val loss."""
    set_seed()
    model = _build_model(vocab_size, num_tags)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=data_prep.PAD_TARGET)

    best_val_loss = float("inf")
    best_epoch = 1
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, optimizer, criterion, train_samples)
        val_loss = _loss_only(model, criterion, val_samples)
        history.append({"epoch": epoch,
                        "train_loss": round(train_loss, 4),
                        "val_loss": round(val_loss, 4)})

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

    return best_epoch, best_val_loss, history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _spans_from_tags(tag_sequence):
    """BIO tags -> set of (start_token, end_token_exclusive, type)."""
    spans = set()
    entity_type = None
    start = None
    for index, tag in enumerate(tag_sequence):
        if tag.startswith("B-"):
            if entity_type is not None:
                spans.add((start, index, entity_type))
            entity_type = tag[2:]
            start = index
        elif tag.startswith("I-") and entity_type == tag[2:]:
            continue
        else:
            if entity_type is not None:
                spans.add((start, index, entity_type))
            entity_type = None
            start = None
    if entity_type is not None:
        spans.add((start, len(tag_sequence), entity_type))
    return spans


def _prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


def evaluate(model, samples, criterion, num_tags):
    """Token metrics + entity-level exact-span metrics."""
    model.eval()
    loss_sum, correct, tokens = 0.0, 0, 0
    predicted_sequences = []
    true_sequences = []

    with torch.no_grad():
        for sample in samples:
            ids, targets = data_prep.make_batch([sample])
            logits = model(ids)
            loss = criterion(logits.reshape(-1, num_tags),
                             targets.reshape(-1))
            count = targets.ne(data_prep.PAD_TARGET).sum().item()
            loss_sum += loss.item() * count
            tokens += count

            predicted_ids = logits.argmax(dim=-1)[0]
            predicted_tags = []
            true_tags = []
            for predicted_id, target_id in zip(predicted_ids, targets[0]):
                target_id = target_id.item()
                if target_id == data_prep.PAD_TARGET:
                    continue
                predicted_tags.append(labels.ID_TO_TAG[predicted_id.item()])
                true_tags.append(labels.ID_TO_TAG[target_id])
            predicted_sequences.append(predicted_tags)
            true_sequences.append(true_tags)
            correct += sum(1 for p, t in zip(predicted_tags, true_tags)
                           if p == t)

    # Per-tag token precision/recall/F1 from flattened pairs
    tag_counts = {tag: [0, 0, 0] for tag in labels.TAGS}  # tp, fp, fn
    for predicted_tags, true_tags in zip(predicted_sequences, true_sequences):
        for predicted, true in zip(predicted_tags, true_tags):
            if predicted == true:
                tag_counts[true][0] += 1
            else:
                tag_counts[predicted][1] += 1
                tag_counts[true][2] += 1

    per_tag = {}
    macro_values = []
    for tag in labels.TAGS:
        if tag == "O":
            continue
        tp, fp, fn = tag_counts[tag]
        precision, recall, f1 = _prf(tp, fp, fn)
        if tp or fp or fn:  # only average over tags present in gold/pred
            macro_values.append(f1)
            per_tag[tag] = {"precision": precision, "recall": recall,
                            "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    macro_f1 = round(sum(macro_values) / len(macro_values), 4) if macro_values else 0.0

    # Entity-level exact span match
    tp = fp = fn = 0
    for predicted_tags, true_tags in zip(predicted_sequences, true_sequences):
        predicted_spans = _spans_from_tags(predicted_tags)
        true_spans = _spans_from_tags(true_tags)
        tp += len(predicted_spans & true_spans)
        fp += len(predicted_spans - true_spans)
        fn += len(true_spans - predicted_spans)
    entity_precision, entity_recall, entity_f1 = _prf(tp, fp, fn)

    return {
        "loss": round(loss_sum / max(tokens, 1), 4),
        "tokens": tokens,
        "token_accuracy": round(correct / max(tokens, 1), 4),
        "token_macro_f1_excl_O": macro_f1,
        "entity_precision": entity_precision,
        "entity_recall": entity_recall,
        "entity_f1": entity_f1,
        "entity_tp": tp, "entity_fp": fp, "entity_fn": fn,
        "per_tag": per_tag,
    }, predicted_sequences, true_sequences


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_artifacts(model, metrics, split, best_epoch, history):
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    torch.save(model.state_dict(),
               os.path.join(config.MODEL_DIR, MODEL_WEIGHTS))

    model_config = {
        "framework": "pytorch",
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "seed": SEED,
        **model.config_dict(),
        "hyperparameters": {
            "embed_dim": EMBED_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "loss": "CrossEntropyLoss(ignore_index=-100)",
            "optimizer": "Adam",
            "epochs_trained": best_epoch,
        },
        "input_format": "LongTensor (batch, length) of Phase 4 token ids",
        "output_format": "FloatTensor (batch, length, num_tags) of logits",
        "label_scheme": "BIO, see tags.json / labels.py",
        "phase4_vocab_file": config.NLP_VOCAB_FILE,
    }
    with open(os.path.join(config.MODEL_DIR, MODEL_CONFIG), "w",
              encoding="utf-8") as fh:
        json.dump(model_config, fh, ensure_ascii=False, indent=2)

    vocab = nlp_preprocess.load_vocab()
    with open(os.path.join(config.MODEL_DIR, MODEL_VOCAB), "w",
              encoding="utf-8") as fh:
        json.dump({"token_to_id": vocab, "size": len(vocab)},
                  fh, ensure_ascii=False, indent=2)

    with open(os.path.join(config.MODEL_DIR, MODEL_TAGS), "w",
              encoding="utf-8") as fh:
        json.dump({
            "tag_to_id": labels.TAG_TO_ID,
            "id_to_tag": {str(index): tag for index, tag in
                          sorted(labels.ID_TO_TAG.items())},
            "scheme": "BIO",
            "entity_types": labels.ENTITY_TYPES,
            "label_rules_version": labels.LABEL_RULES_VERSION,
            "label_generation": "See labels.py module docstring "
                                "(deterministic regex rules over the "
                                "clinical-note body).",
        }, fh, ensure_ascii=False, indent=2)

    with open(os.path.join(config.MODEL_DIR, MODEL_SPLIT), "w",
              encoding="utf-8") as fh:
        json.dump({name: [s["document_id"] for s in group]
                   for name, group in split.items()}, fh, indent=2)

    payload = {
        "seed": SEED,
        "split_sizes": {name: len(group) for name, group in split.items()},
        "phase_a": {"best_epoch": best_epoch,
                    "epochs_history": history},
        "final_model": metrics,
        "limitations": LIMITATIONS,
    }
    with open(os.path.join(config.MODEL_DIR, MODEL_METRICS), "w",
              encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    started = time.time()
    print("=== Phase 5: LSTM training ===")

    samples = data_prep.build_samples()
    train_samples, val_samples, test_samples = data_prep.split_samples(samples)
    split = {"train": train_samples, "val": val_samples, "test": test_samples}
    print(f"split: train={len(train_samples)} val={len(val_samples)} "
          f"test={len(test_samples)}")

    vocab = nlp_preprocess.load_vocab()
    vocab_size = len(vocab)
    num_tags = len(labels.TAGS)

    # Phase A: epoch selection on val
    best_epoch, best_val_loss, history = select_epoch_count(
        train_samples, val_samples, vocab_size, num_tags)
    print(f"phase A: best epoch = {best_epoch} "
          f"(val loss {best_val_loss:.4f}, {len(history)} epochs run)")

    # Phase B: fresh model on train+val for best_epoch epochs, test once
    set_seed()
    model = _build_model(vocab_size, num_tags)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=data_prep.PAD_TARGET)
    final_train = train_samples + val_samples
    for _ in range(best_epoch):
        train_one_epoch(model, optimizer, criterion, final_train)

    metrics = {}
    for name, group in (("train", final_train),
                        ("val", val_samples),
                        ("test", test_samples)):
        result, _, _ = evaluate(model, group, criterion, num_tags)
        metrics[name] = result
        print(f"{name:>5}: loss={result['loss']:.4f} "
              f"token_acc={result['token_accuracy']:.4f} "
              f"macro_f1={result['token_macro_f1_excl_O']:.4f} "
              f"entity_P/R/F1={result['entity_precision']:.4f}/"
              f"{result['entity_recall']:.4f}/{result['entity_f1']:.4f}")

    save_artifacts(model, metrics, split, best_epoch, history)
    print()
    print("artifacts written:")
    for filename in (MODEL_WEIGHTS, MODEL_CONFIG, MODEL_VOCAB,
                     MODEL_TAGS, MODEL_METRICS, MODEL_SPLIT):
        print(f"  {os.path.join(config.MODEL_DIR, filename)}")
    print(f"elapsed: {time.time() - started:.1f}s")
    return metrics


if __name__ == "__main__":
    main()
