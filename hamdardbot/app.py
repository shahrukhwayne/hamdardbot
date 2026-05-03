"""
app.py — Flask Backend for Jamia Hamdard Chatbot.

Endpoints:
  POST /api/chat    — Main chat endpoint
  GET  /api/health  — System health check
  GET  /api/stats   — Runtime statistics

Run:
  python app.py
  or with gunicorn: gunicorn app:app -w 2 -b 0.0.0.0:5000
"""

import os
import time
import json
import logging
import traceback
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import llm_handler
from intent_handler import get_handler

# ─────────────────────────────────────────────
#  LOGGING SETUP
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("chatbot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("app")


# ─────────────────────────────────────────────
#  FLASK APP
# ─────────────────────────────────────────────

app = Flask(__name__, static_folder="static")
CORS(app)  # Allow cross-origin requests from frontend

# ── Runtime Stats ──
_stats = {
    "total_queries": 0,
    "static_responses": 0,
    "dynamic_responses": 0,
    "llm_responses": 0,
    "nlp_keyword_responses": 0,
    "errors": 0,
    "start_time": datetime.now().isoformat(),
}


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the chat UI."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint.

    Request JSON:
        { "message": "your question here" }

    Response JSON:
        {
          "response": "...",
          "intent": "...",
          "confidence": 0.92,
          "response_type": "static | dynamic | nlp_keyword | llm",
          "debug_trace": [...]
        }
    """
    t_start = time.time()

    try:
        data = request.get_json(force=True, silent=True) or {}
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"error": "Message cannot be empty."}), 400

        if len(user_message) > 500:
            return jsonify({"error": "Message too long (max 500 chars)."}), 400

        logger.info(f"[Chat] Incoming: '{user_message}'")

        # Always use LLM (force detailed answers)
        handler = get_handler()

        if not handler.is_ready():
            response_text = llm_handler.llm_fallback(user_message)

            result = {
                "response": response_text,
                "intent": "llm_direct",
                "confidence": 0.0,
                "response_type": "llm",
                "debug_trace": ["Model not trained — using LLM"],
            }

        else:
            try:
                result = handler.handle(user_message)
            except Exception as e:
                print("🔥 HANDLER ERROR:", e)

                response_text = llm_handler.llm_fallback(user_message)

                result = {
                    "response": response_text,
                    "intent": "llm_backup",
                    "confidence": 0.0,
                    "response_type": "llm",
                    "debug_trace": ["Fallback after handler crash"],
                }

        # Update stats
        _stats["total_queries"] += 1
        rt = result.get("response_type", "static")
        if rt == "static":
            _stats["static_responses"] += 1
        elif rt == "dynamic":
            _stats["dynamic_responses"] += 1
        elif rt == "llm":
            _stats["llm_responses"] += 1
        elif rt == "nlp_keyword":
            _stats["nlp_keyword_responses"] += 1

        elapsed_ms = round((time.time() - t_start) * 1000, 2)
        result["response_time_ms"] = elapsed_ms

        logger.info(
            f"[Chat] intent={result['intent']} type={result['response_type']} "
            f"conf={result['confidence']} time={elapsed_ms}ms"
        )

        return jsonify(result)

    except Exception as e:
        _stats["errors"] += 1
        logger.error(f"[Chat Error] {e}\n{traceback.format_exc()}")
        return jsonify({
            "response": "I encountered an unexpected error. Please try again.",
            "intent": "error",
            "confidence": 0.0,
            "response_type": "error",
            "debug_trace": [str(e)],
        }), 500


@app.route("/api/health", methods=["GET"])
def health():
    """System health check endpoint."""
    handler = get_handler()
    model_trained = handler.is_ready()

    health_data = {
        "status": "healthy" if model_trained else "degraded",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_trained,
        "llm_mode": llm_handler.get_mode(),
        "vocab_size": len(handler.vocabulary) if model_trained else 0,
        "num_intents": len(handler.tags) if model_trained else 0,
        "grok_api_enabled": bool(os.environ.get("GROK_API_KEY")),
    }

    if not model_trained:
        health_data["warning"] = "Model not trained. Run: python train.py"

    return jsonify(health_data), 200


@app.route("/api/stats", methods=["GET"])
def stats():
    """Runtime statistics endpoint."""
    from scraper import _cache as scraper_cache

    return jsonify({
        **_stats,
        "uptime_seconds": round(
            (datetime.now() - datetime.fromisoformat(_stats["start_time"])).total_seconds()
        ),
        "scraper_cache_entries": len(scraper_cache),
    })


@app.route("/api/intents", methods=["GET"])
def list_intents():
    """List all available intents (for debugging / admin)."""
    from intents import INTENTS
    return jsonify([
        {"tag": i["tag"], "patterns": len(i["patterns"]), "dynamic": i.get("dynamic", False)}
        for i in INTENTS
    ])


# ─────────────────────────────────────────────
#  ERROR HANDLERS
# ─────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed."}), 405


# ─────────────────────────────────────────────
#  STARTUP
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  Jamia Hamdard Chatbot — Flask Server Starting")
    logger.info("=" * 60)

    handler = get_handler()
    if not handler.is_ready():
        logger.warning("⚠️  Model not found. Please run: python train.py")
    else:
        logger.info(f"✅ Model loaded. {len(handler.tags)} intents, {len(handler.vocabulary)} vocab words.")

    logger.info(f"LLM Mode: {llm_handler.get_mode()}")
    logger.info("Starting Flask server on http://0.0.0.0:5000")

    app.run(host="0.0.0.0", port=5000, debug=False)
