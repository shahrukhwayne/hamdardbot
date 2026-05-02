"""
model.py — Neural Network (MLP) from scratch using NumPy only.
No TensorFlow, no PyTorch. Pure NumPy implementation.

Architecture: Input → Hidden1 → Hidden2 → Output (Softmax)
Training: Mini-batch gradient descent with backpropagation
"""

import numpy as np
import json
import pickle
import os
import re
from collections import Counter


# ─────────────────────────────────────────────
#  TEXT PROCESSING
# ─────────────────────────────────────────────

def tokenize(sentence: str) -> list:
    """Lowercase, remove punctuation, split into tokens."""
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-z0-9\s]", "", sentence)
    return sentence.split()


def stem(word: str) -> str:
    """
    Lightweight suffix-stripping stemmer (Porter-inspired).
    Strips common English suffixes without NLTK.
    """
    suffixes = ["ing", "tion", "ness", "ment", "ly", "er", "est", "ed", "es", "s"]
    for suffix in sorted(suffixes, key=len, reverse=True):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def build_vocabulary(all_patterns: list) -> list:
    """Build sorted vocabulary from all training patterns."""
    words = []
    for pattern in all_patterns:
        tokens = tokenize(pattern)
        words.extend([stem(t) for t in tokens])
    vocab = sorted(set(words))
    return vocab


def bag_of_words(sentence: str, vocabulary: list) -> np.ndarray:
    """Convert sentence to bag-of-words vector."""
    tokens = [stem(t) for t in tokenize(sentence)]
    bag = np.zeros(len(vocabulary), dtype=np.float32)
    for i, word in enumerate(vocabulary):
        if word in tokens:
            bag[i] = 1.0
    return bag


# ─────────────────────────────────────────────
#  ACTIVATIONS
# ─────────────────────────────────────────────

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def relu_deriv(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(np.float32)


def softmax(z: np.ndarray) -> np.ndarray:
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    n = y_pred.shape[0]
    log_likelihood = -np.log(y_pred[range(n), y_true.argmax(axis=1)] + 1e-9)
    return np.mean(log_likelihood)


# ─────────────────────────────────────────────
#  MLP MODEL CLASS
# ─────────────────────────────────────────────

class MLPClassifier:
    """
    Multi-Layer Perceptron (MLP) Intent Classifier.
    Architecture: Input → ReLU → Dropout → ReLU → Dropout → Softmax
    """

    def __init__(self, input_size: int, hidden_sizes: list, output_size: int,
                 learning_rate: float = 0.01, dropout_rate: float = 0.3):
        self.lr = learning_rate
        self.dropout_rate = dropout_rate
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = []
        self.biases = []
        self._init_weights()

    def _init_weights(self):
        """He initialization for ReLU networks."""
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            w = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, fan_out))
            self.weights.append(w)
            self.biases.append(b)

    def _dropout(self, x: np.ndarray, training: bool) -> tuple:
        if not training or self.dropout_rate == 0:
            mask = np.ones_like(x)
            return x, mask
        mask = (np.random.rand(*x.shape) > self.dropout_rate).astype(np.float32)
        return x * mask / (1.0 - self.dropout_rate), mask

    def forward(self, X: np.ndarray, training: bool = False) -> tuple:
        """Forward pass. Returns output probabilities and cache for backprop."""
        cache = {"X": X}
        a = X
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            cache[f"z{i}"] = z
            if i < len(self.weights) - 1:
                a = relu(z)
                a, mask = self._dropout(a, training)
                cache[f"a{i}"] = a
                cache[f"mask{i}"] = mask
            else:
                a = softmax(z)
                cache[f"out"] = a
        return a, cache

    def backward(self, cache: dict, y_true: np.ndarray):
        """Backpropagation — compute gradients and update weights."""
        m = cache["X"].shape[0]
        out = cache["out"]
        n_layers = len(self.weights)

        # Output layer gradient
        dz = out - y_true  # (m, output_size)

        grads_w = [None] * n_layers
        grads_b = [None] * n_layers

        for i in reversed(range(n_layers)):
            a_prev = cache.get(f"a{i-1}", cache["X"]) if i > 0 else cache["X"]
            grads_w[i] = (a_prev.T @ dz) / m
            grads_b[i] = np.mean(dz, axis=0, keepdims=True)

            if i > 0:
                da_prev = dz @ self.weights[i].T
                # Apply dropout mask
                mask = cache.get(f"mask{i-1}", np.ones_like(da_prev))
                da_prev = da_prev * mask / (1.0 - self.dropout_rate + 1e-9)
                dz = da_prev * relu_deriv(cache[f"z{i-1}"])

        # Update weights with gradient descent
        for i in range(n_layers):
            self.weights[i] -= self.lr * grads_w[i]
            self.biases[i] -= self.lr * grads_b[i]

    def predict(self, X: np.ndarray) -> tuple:
        """Return predicted class index and confidence score."""
        probs, _ = self.forward(X, training=False)
        return np.argmax(probs, axis=1), np.max(probs, axis=1)

    def save(self, weights_path: str = "model/weights.pkl",
             meta_path: str = "model/meta.json"):
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        with open(weights_path, "wb") as f:
            pickle.dump({"weights": self.weights, "biases": self.biases}, f)
        meta = {
            "layer_sizes": self.layer_sizes,
            "learning_rate": self.lr,
            "dropout_rate": self.dropout_rate,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[Model] Saved weights to {weights_path} and meta to {meta_path}")

    @classmethod
    def load(cls, weights_path: str = "model/weights.pkl",
             meta_path: str = "model/meta.json") -> "MLPClassifier":
        with open(meta_path, "r") as f:
            meta = json.load(f)
        sizes = meta["layer_sizes"]
        model = cls(
            input_size=sizes[0],
            hidden_sizes=sizes[1:-1],
            output_size=sizes[-1],
            learning_rate=meta.get("learning_rate", 0.01),
            dropout_rate=meta.get("dropout_rate", 0.3),
        )
        with open(weights_path, "rb") as f:
            data = pickle.load(f)
        model.weights = data["weights"]
        model.biases = data["biases"]
        print("[Model] Loaded successfully.")
        return model
