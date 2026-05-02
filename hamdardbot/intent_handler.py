"""
intent_handler.py — Hybrid Decision Engine for the Jamia Hamdard Chatbot.

Decision Flow:
  1. Predict intent via ML model
  2. Validate context (confidence + context mismatch check)
  3. If dynamic intent → call scraper
  4. If static intent → return ML response
  5. If confidence < threshold → NLP keyword fallback
  6. If still unclear → LLM fallback
"""

import pickle
import random
import logging
import re
from typing import Optional
import numpy as np

from model import MLPClassifier, bag_of_words
from intents import INTENTS
from scraper import (
    get_news, get_admissions, get_notices,
    format_scraped_response
)
import llm_handler

logger = logging.getLogger("intent_handler")

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

ML_CONFIDENCE_THRESHOLD = 0.75   # below this → try NLP keyword fallback
NLP_CONFIDENCE_THRESHOLD = 0.30  # below this → LLM fallback
MODEL_DIR = "model"

# Dynamic intents require live scraping
DYNAMIC_INTENTS = {"admission", "news", "notices"}

# Context mismatch: if top-predicted intent is far from query content,
# these keyword sets are used to override or reduce confidence
CONTEXT_KEYWORDS = {
    "greeting":      ["hello", "hi", "hey", "salam", "good morning", "good evening"],
    "farewell":      ["bye", "goodbye", "see you", "take care"],
    "admission":     ["admission", "apply", "application", "eligibility", "enroll"],
    "fees":          ["fee", "fees", "cost", "price", "payment", "tuition"],
    "hostel":        ["hostel", "accommodation", "room", "stay", "dormitory"],
    "courses":       ["course", "program", "degree", "study", "department"],
    "placements":    ["placement", "job", "recruit", "salary", "career"],
    "scholarships":  ["scholarship", "financial aid", "waiver", "merit", "bursary"],
    "news":          ["news", "latest", "update", "event", "announcement"],
    "notices":       ["notice", "circular", "official", "exam notice"],
    "contact":       ["contact", "phone", "email", "address", "location"],
    "bot_identity":  ["who are you", "what are you", "bot", "ai", "robot"],
    "rankings":      ["rank", "ranking", "nirf", "naac", "accreditation"],
}


# ─────────────────────────────────────────────
#  MODEL LOADER
# ─────────────────────────────────────────────

class IntentHandler:
    """
    Central handler for the hybrid intent classification pipeline.
    Loads the trained MLP model and routes queries appropriately.
    """

    def __init__(self):
        self.model: Optional[MLPClassifier] = None
        self.vocabulary: list = []
        self.tags: list = []
        self.tag_index: dict = {}
        self.tag_responses: dict = {}
        self.tag_dynamic: dict = {}
        self._build_response_map()
        self._load_model()
        logger.info("[IntentHandler] Initialized.")

    def _build_response_map(self):
        """Build lookup dicts from INTENTS list."""
        for intent in INTENTS:
            tag = intent["tag"]
            self.tag_responses[tag] = intent["responses"]
            self.tag_dynamic[tag] = intent.get("dynamic", False)
            
    def _handle_dynamic(self, intent: str):
        try:
            if intent == "admission":
                data = get_admissions()
            elif intent == "news":
                data = get_news()
            elif intent == "notices":
                data = get_notices()
            else:
                return {"response": "No dynamic handler found."}

            return format_scraped_response(data)

        except Exception as e:
            return {"response": f"Error fetching live data: {str(e)}"}

    def _load_model(self):
        """Load trained MLP model and vocabulary from disk."""
        try:
            with open(f"{MODEL_DIR}/vocab.pkl", "rb") as f:
                vocab_data = pickle.load(f)
            self.vocabulary = vocab_data["vocabulary"]
            self.tags = vocab_data["tags"]
            self.tag_index = vocab_data["tag_index"]
            self.model = MLPClassifier.load(
                weights_path=f"{MODEL_DIR}/weights.pkl",
                meta_path=f"{MODEL_DIR}/meta.json",
            )
            logger.info(f"[IntentHandler] Model loaded. Tags: {len(self.tags)}, Vocab: {len(self.vocabulary)}")
        except FileNotFoundError:
            logger.warning("[IntentHandler] Model not found. Run train.py first.")
            self.model = None

    def is_ready(self) -> bool:
        return self.model is not None

    # ─────────────────────────────────────────
    #  CORE PREDICTION
    # ─────────────────────────────────────────

    def _ml_predict(self, query: str) -> tuple:
        """Run ML prediction. Returns (tag, confidence)."""
        bow = bag_of_words(query, self.vocabulary).reshape(1, -1)
        pred_idx, conf = self.model.predict(bow)
        tag = self.tags[pred_idx[0]]
        return tag, float(conf[0])

    def _context_validate(self, query: str, predicted_tag: str, confidence: float) -> tuple:
        """
        Context mismatch detection.
        If the query contains keywords clearly belonging to a DIFFERENT intent,
        penalise the predicted confidence or override it.
        Returns (validated_tag, adjusted_confidence).
        """
        query_lower = query.lower()

        # Check if any context keywords for a different tag appear strongly
        keyword_votes = {}
        for tag, keywords in CONTEXT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                keyword_votes[tag] = score

        if not keyword_votes:
            return predicted_tag, confidence

        top_keyword_tag = max(keyword_votes, key=keyword_votes.get)
        top_score = keyword_votes[top_keyword_tag]

        # If keyword evidence strongly points elsewhere, penalise ML confidence
        if top_keyword_tag != predicted_tag and top_score >= 2:
            adjusted_conf = confidence * 0.5
            logger.debug(
                f"[Context] Mismatch: ML={predicted_tag}({confidence:.2f}) "
                f"Keywords={top_keyword_tag}({top_score}) → adjusted conf={adjusted_conf:.2f}"
            )
            return predicted_tag, adjusted_conf

        # If keyword match confirms prediction, slightly boost confidence
        if top_keyword_tag == predicted_tag:
            boosted = min(0.99, confidence * 1.1)
            return predicted_tag, boosted

        return predicted_tag, confidence

    def _nlp_keyword_fallback(self, query: str) -> tuple:
        """
        Rule-based NLP keyword matching as secondary classifier.
        Returns (tag, confidence) or (None, 0.0).
        """
        query_lower = query.lower()
        best_tag = None
        best_score = 0

        for tag, keywords in CONTEXT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > best_score:
                best_score = score
                best_tag = tag

        if best_tag and best_score > 0:
            nlp_conf = min(0.45, 0.15 * best_score)
            logger.info(f"[NLP Fallback] tag={best_tag}, score={best_score}, conf={nlp_conf:.2f}")
            return best_tag, nlp_conf

        return None, 0.0

    # ─────────────────────────────────────────
    #  RESPONSE GENERATION
    # ─────────────────────────────────────────

    def _get_static_response(self, tag: str) -> str:
        responses = self.tag_responses.get(tag, [])
        if responses:
            return random.choice(responses)
        return "I'm not sure I have information about that. Please visit jamiahamdard.ac.in for more details."

    def _get_dynamic_response(self, tag: str) -> str:
        """Call scraper based on intent tag."""
        try:
            if tag == "admission":
                data = get_admissions()
                return format_scraped_response(data, "Latest Admission Information")
            elif tag == "news":
                data = get_news()
                return format_scraped_response(data, "Latest News & Events")
            elif tag == "notices":
                data = get_notices()
                return format_scraped_response(data, "Official Notices")
        except Exception as e:
            logger.error(f"[Scraper Error] tag={tag}: {e}")
        # Fallback to static
        return self._get_static_response(tag)

    # ─────────────────────────────────────────
    #  MAIN HANDLER
    # ─────────────────────────────────────────

    def handle(self, query: str) -> dict:
        debug = []

        # ✅ 1. Quick rules (fast responses)
        q = query.lower().strip()
        if q in ["hi", "hello", "hey", "hii"]:
            return {
                "response": "Hello! How can I help you with Jamia Hamdard?",
                "intent": "greeting",
                "confidence": 1.0,
                "response_type": "static",
                "debug_trace": ["Quick greeting rule"]
            }

        # ✅ 2. ML prediction
        intent, confidence = self._ml_predict(query)
        debug.append(f"ML: {intent} ({confidence:.2f})")

        # ✅ 3. Context validation
        intent, confidence = self._context_validate(query, intent, confidence)
        debug.append(f"After context: {intent} ({confidence:.2f})")

        # 🔥 4. DECISION ENGINE

        # ✅ HIGH confidence → normal flow
        if confidence >= 0.80:
            if intent in DYNAMIC_INTENTS:
                data = self._handle_dynamic(intent)
                return {
                    "response": data,
                    "intent": intent,
                    "confidence": confidence,
                    "response_type": "dynamic",
                    "debug_trace": debug
                }

            return {
                "response": random.choice(self.tag_responses[intent]),
                "intent": intent,
                "confidence": confidence,
                "response_type": "static",
                "debug_trace": debug
            }

        # ⚡ MID confidence → LLM + ML mix
        if 0.40 <= confidence < 0.80:
            debug.append("MID confidence → LLM assist")

            llm_res = llm_handler.llm_fallback(query)

            if llm_res:
                return {
                    "response": llm_res,
                    "intent": "llm_assist",
                    "confidence": confidence,
                    "response_type": "llm",
                    "debug_trace": debug
                }

        # 🔥 LOW confidence → DIRECT LLM
        debug.append("LOW confidence → LLM")

        llm_res = llm_handler.llm_fallback(query)

        return {
            "response": llm_res,
            "intent": "llm",
            "confidence": confidence,
            "response_type": "llm",
            "debug_trace": debug
        }

# ─────────────────────────────────────────────
#  SINGLETON INSTANCE
# ─────────────────────────────────────────────

_handler_instance: Optional[IntentHandler] = None


def get_handler() -> IntentHandler:
    global _handler_instance
    if _handler_instance is None:
        _handler_instance = IntentHandler()
    return _handler_instance
