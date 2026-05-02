# 🌿 HamdardBot — Jamia Hamdard University AI Chatbot

A production-grade **hybrid AI chatbot** combining Machine Learning, Real-time Web Scraping, and LLM fallback, built specifically for Jamia Hamdard University.

---

## 📁 Project Structure

```
jamia_chatbot/
├── app.py              # Flask backend (API + static serving)
├── model.py            # NumPy MLP neural network from scratch
├── train.py            # Training script
├── intents.py          # 27 intents with patterns & responses
├── scraper.py          # Real-time web scraper (TTL cache + retry)
├── intent_handler.py   # Hybrid decision engine
├── llm_handler.py      # LLM fallback (Grok API or rule-based NLP)
├── requirements.txt    # Python dependencies
├── static/
│   └── index.html      # Chat UI (ChatGPT-style)
└── model/              # Created after training
    ├── weights.pkl     # Neural network weights
    ├── meta.json       # Model architecture metadata
    ├── vocab.pkl       # Vocabulary + tag mappings
    └── training_summary.json
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
```
This will:
- Build vocabulary from 27 intents
- Train a NumPy MLP (no TensorFlow/PyTorch)
- Save model to `model/` directory

### 3. Start the server
```bash
python app.py
```
Open your browser at: **http://localhost:5000**

---

## 🔑 Optional: Grok API (LLM Mode)

Set your xAI Grok API key as an environment variable:

```bash
# Linux / Mac
export GROK_API_KEY=your_key_here

# Windows
set GROK_API_KEY=your_key_here
```

Without the key, the system uses the **rule-based NLP fallback** automatically.

---

## 🧠 System Architecture

```
User Query
    │
    ▼
[ML Model] ─── confidence ≥ 0.50 ──► [Dynamic?] ──Yes──► [Web Scraper] ──► Response
    │                                     │ No
    │                                     └──────────────► [Static Response]
    │
    └── confidence < 0.50 ──► [NLP Keywords] ─── match? ──► Response
                                    │ No
                                    └──────────────► [LLM Fallback] ──► Response
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | Main chat endpoint |
| GET | `/api/health` | System health check |
| GET | `/api/stats` | Runtime statistics |
| GET | `/api/intents` | List all intents |

### Chat Request Example
```json
POST /api/chat
{ "message": "How do I apply for admission?" }
```

### Response
```json
{
  "response": "Fetching latest admission info...",
  "intent": "admission",
  "confidence": 0.94,
  "response_type": "dynamic",
  "debug_trace": ["ML predicted: admission (94.2%)", "Dynamic intent → scraping"],
  "response_time_ms": 342
}
```

---

## 🎯 Supported Intents (27 total)

greeting, goodbye, bot_identity, courses, admission *(dynamic)*, fees, hostel, scholarships, placements, rankings, contact, news *(dynamic)*, notices *(dynamic)*, library, sports, faculty, research, transport, canteen, exams, clubs, international, pharmacy, medicine, management, thanks, confused

---

## 🛠 Production Deployment

```bash
# With Gunicorn
gunicorn app:app -w 4 -b 0.0.0.0:5000

# With Docker (create Dockerfile separately)
docker build -t hamdardbot .
docker run -p 5000:5000 -e GROK_API_KEY=your_key hamdardbot
```

---

## 📊 Tech Stack

| Component | Technology |
|-----------|------------|
| ML Model | NumPy MLP (from scratch) |
| Web Scraping | requests + BeautifulSoup4 |
| Backend API | Flask + Flask-CORS |
| LLM Fallback | xAI Grok API / Rule-based NLP |
| Frontend | Vanilla HTML/CSS/JS |
| Model Storage | pickle + JSON |

---

*Built for Jamia Hamdard University, Hamdard Nagar, New Delhi - 110062*
