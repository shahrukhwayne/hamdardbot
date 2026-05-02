import os
import re
import logging
import json
import random
from typing import Optional

import requests

logger = logging.getLogger("llm_handler")

# =========================
# CONFIG
# =========================

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "meta-llama/llama-3-8b-instruct"
MAX_TOKENS = 700

SYSTEM_PROMPT = (
    "You are HamdardBot, an expert AI assistant for Jamia Hamdard University.\n"
    "Give detailed, informative, and engaging answers like a university brochure.\n"
    "Always include:\n"
    "- Overview of the university\n"
    "- Courses and faculties\n"
    "- Campus facilities\n"
    "- Placements and reputation\n"
    "- Keep it informative and structured\n"
    "Do NOT give short answers.\n"
    "Do NOT say 'ask me more'.\n"
    "Answer like a professional university guide."
)

# =========================
# LLM CALL (FREE)
# =========================

def _call_llm(query: str) -> Optional[str]:
    api_key = "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"

    if not api_key:
        logger.warning("[LLM] No API key found.")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{query}\n\nGive a detailed structured answer."}
        ],
        "temperature": 0.7,
        "max_tokens": MAX_TOKENS
    }

    try:
        res = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=10)
        res.raise_for_status()
        data = res.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("🔥 LLM ERROR:", e)
        return None


# =========================
# RULE BASED FALLBACK
# =========================

RULES = {
    r"admission|apply": "Visit jamiahamdard.edu to apply for admission. Different courses have different criteria.",
    r"fees|fee|cost": "Fees vary by course. Pharmacy approx ₹1L/year. Check official website.",
    r"hostel": "Hostel facility available with mess, WiFi, and security.",
    r"placement|job": "Strong placements in Pharmacy and Management.",
}

FALLBACK = [
    "Ask me about admissions, courses, fees, hostel, or placements.",
    "Visit jamiahamdard.edu for detailed info.",
]

def _rule_based(query: str) -> str:
    q = query.lower()

    for pattern, response in RULES.items():
        if re.search(pattern, q):
            return response

    return random.choice(FALLBACK)


# =========================
# MAIN FUNCTION
# =========================

def llm_fallback(query: str) -> str:
    logger.info("[LLM] Trying FREE LLM...")

    response = _call_llm(query)

    if response:
        return response

    logger.warning("[LLM] Falling back to rule-based NLP")

    return _rule_based(query)


def get_mode() -> str:
   return "openrouter_llm"
