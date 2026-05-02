"""
train.py — Training script for the Jamia Hamdard Intent Classifier.

Workflow:
  1. Load intents from intents.py
  2. Build vocabulary and one-hot labels
  3. Train MLP with mini-batch gradient descent
  4. Evaluate on training set
  5. Save model/weights.pkl and model/meta.json
  6. Save vocabulary + tag mapping to model/vocab.pkl

Run: python train.py
"""

import numpy as np
import pickle
import json
import os
import sys
import time

from intents import INTENTS
from model import (
    MLPClassifier,
    build_vocabulary,
    bag_of_words,
    cross_entropy_loss,
    tokenize,
    stem,
)


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

HIDDEN_SIZES = [128, 64]
LEARNING_RATE = 0.005
EPOCHS = 1000
BATCH_SIZE = 16
DROPOUT_RATE = 0.3
PRINT_EVERY = 100
MODEL_DIR = "model"


# ─────────────────────────────────────────────
#  DATA PREPARATION
# ─────────────────────────────────────────────

def prepare_data(intents: list) -> tuple:
    """
    Convert intents into training (X, y) arrays.
    Returns:
        X          — shape (N, vocab_size)
        y          — shape (N, num_classes) one-hot
        vocabulary — sorted list of stemmed tokens
        tags       — list of intent tag strings
        tag_index  — dict mapping tag → int index
    """
    all_patterns = []
    all_tags = []

    tags = []
    for intent in intents:
        tag = intent["tag"]
        if tag not in tags:
            tags.append(tag)
        for pattern in intent["patterns"]:
            all_patterns.append(pattern)
            all_tags.append(tag)

    vocabulary = build_vocabulary(all_patterns)
    tag_index = {tag: i for i, tag in enumerate(tags)}

    X = np.array([bag_of_words(p, vocabulary) for p in all_patterns], dtype=np.float32)
    y_indices = np.array([tag_index[t] for t in all_tags])
    y = np.eye(len(tags), dtype=np.float32)[y_indices]

    print(f"[Train] Vocab size     : {len(vocabulary)}")
    print(f"[Train] Num intents    : {len(tags)}")
    print(f"[Train] Training samples: {len(X)}")

    return X, y, vocabulary, tags, tag_index


# ─────────────────────────────────────────────
#  MINI-BATCH GENERATOR
# ─────────────────────────────────────────────

def batch_generator(X: np.ndarray, y: np.ndarray, batch_size: int):
    """Yield shuffled mini-batches."""
    indices = np.random.permutation(len(X))
    X_shuffled, y_shuffled = X[indices], y[indices]
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        yield X_shuffled[start:end], y_shuffled[start:end]


# ─────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────

def train(intents: list):
    print("\n" + "=" * 60)
    print("  Jamia Hamdard Chatbot — Training")
    print("=" * 60)

    X, y, vocabulary, tags, tag_index = prepare_data(intents)

    model = MLPClassifier(
        input_size=len(vocabulary),
        hidden_sizes=HIDDEN_SIZES,
        output_size=len(tags),
        learning_rate=LEARNING_RATE,
        dropout_rate=DROPOUT_RATE,
    )

    start_time = time.time()
    best_loss = float("inf")
    patience = 150
    no_improve_count = 0

    for epoch in range(1, EPOCHS + 1):
        epoch_losses = []

        for X_batch, y_batch in batch_generator(X, y, BATCH_SIZE):
            output, cache = model.forward(X_batch, training=True)
            loss = cross_entropy_loss(output, y_batch)
            model.backward(cache, y_batch)
            epoch_losses.append(loss)

        avg_loss = np.mean(epoch_losses)

        # Early stopping
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            no_improve_count = 0
        else:
            no_improve_count += 1

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            elapsed = time.time() - start_time
            # Compute training accuracy
            preds, _ = model.predict(X)
            true_labels = np.argmax(y, axis=1)
            acc = np.mean(preds == true_labels) * 100
            print(
                f"  Epoch {epoch:>5}/{EPOCHS} | "
                f"Loss: {avg_loss:.4f} | "
                f"Acc: {acc:.1f}% | "
                f"Time: {elapsed:.1f}s"
            )

        if no_improve_count >= patience:
            print(f"\n[Train] Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    elapsed = time.time() - start_time
    print(f"\n[Train] Training complete in {elapsed:.2f}s")

    # Final evaluation
    preds, confidences = model.predict(X)
    true_labels = np.argmax(y, axis=1)
    final_acc = np.mean(preds == true_labels) * 100
    print(f"[Train] Final Training Accuracy: {final_acc:.2f}%")
    print(f"[Train] Final Loss: {best_loss:.4f}")

    # Save model artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(
        weights_path=f"{MODEL_DIR}/weights.pkl",
        meta_path=f"{MODEL_DIR}/meta.json",
    )

    vocab_data = {
        "vocabulary": vocabulary,
        "tags": tags,
        "tag_index": tag_index,
    }
    with open(f"{MODEL_DIR}/vocab.pkl", "wb") as f:
        pickle.dump(vocab_data, f)
    print(f"[Train] Vocabulary saved to {MODEL_DIR}/vocab.pkl")

    # Save training summary
    summary = {
        "vocab_size": len(vocabulary),
        "num_intents": len(tags),
        "num_samples": len(X),
        "final_accuracy": round(final_acc, 2),
        "final_loss": round(float(best_loss), 4),
        "hidden_sizes": HIDDEN_SIZES,
        "learning_rate": LEARNING_RATE,
        "epochs_run": epoch,
        "tags": tags,
    }
    with open(f"{MODEL_DIR}/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Train] All artifacts saved to /{MODEL_DIR}/")
    print("=" * 60)
    return model, vocabulary, tags, tag_index


# ─────────────────────────────────────────────
#  QUICK TEST
# ─────────────────────────────────────────────

def quick_test(model, vocabulary, tags):
    print("\n[Test] Quick inference test:")
    test_queries = [
        "hello there",
        "how to apply for admission",
        "what is the fee structure",
        "tell me about hostel",
        "who are you",
        "latest news",
    ]
    for query in test_queries:
        from model import bag_of_words
        bow = bag_of_words(query, vocabulary).reshape(1, -1)
        pred, conf = model.predict(bow)
        tag = tags[pred[0]]
        print(f"  '{query}' → {tag} ({conf[0]*100:.1f}%)")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    model, vocabulary, tags, tag_index = train(INTENTS)
    quick_test(model, vocabulary, tags)
    print("\n✅ Training complete! You can now run: python app.py\n")
