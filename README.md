# Ammamma

An AI grandmother who keeps scam callers on the line. Gnani Prisma (STT) + Timbre (TTS) on a Pipecat phone pipeline; Groq LLM until self-hosted Evon is up.

## Setup

```bash
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
cp .env.example .env   # fill GNANI_API_KEY and GROQ_API_KEY
.venv/Scripts/python test_bot.py
```

## Run

```bash
.venv/Scripts/python bot.py                                   # browser: http://localhost:7860/client
.venv/Scripts/python bot.py -t twilio -x <name>.ngrok.app     # phone (point the number's voice webhook at https://<name>.ngrok.app/)
.venv/Scripts/python bot.py -t exotel -x <name>.ngrok.app     # phone via Exotel (Indian numbers)
```

Each call's transcript lands in `calls/` (gitignored).

## Switching to Evon

Serve Evon with vLLM (see its Hugging Face card), then set `EVON_BASE_URL=http://<host>:8000/v1` in `.env`. Thinking is turned off per request to keep replies fast.

## Tuning

The character lives in `persona.md`; edit it, no code changes needed. Language and voice are `AMMAMMA_LANG` / `AMMAMMA_VOICE` in `.env`.
