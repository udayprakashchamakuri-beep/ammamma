"""Run: .venv/Scripts/python test_bot.py"""

import json
import tempfile
from pathlib import Path

from gnani.tts.client import TIMBRE_V25_VOICES
from pipecat.services.google.llm import GoogleLLMService

from bot import LANGUAGES, SWITCH_LANGUAGE, make_llm, save_call, system_prompt

prompt = system_prompt("te-IN")
assert "start in Telugu" in prompt and "{language}" not in prompt

# Every language must have a real Timbre v2.5 voice, and the tool must offer exactly these.
assert all(voice in TIMBRE_V25_VOICES for _, voice in LANGUAGES.values())
assert SWITCH_LANGUAGE.properties["code"]["enum"] == list(LANGUAGES)

gemini = make_llm({"GEMINI_API_KEY": "x"}, prompt)
assert type(gemini) is GoogleLLMService
assert gemini._settings.thinking.thinking_budget == 0

evon = make_llm({"EVON_BASE_URL": "http://localhost:8000/v1", "GEMINI_API_KEY": "x"}, prompt)
assert "localhost" in str(evon._client.base_url)
assert evon._settings.extra["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False

try:
    make_llm({}, prompt)
    raise AssertionError("expected missing-key error")
except RuntimeError:
    pass

# Concurrent reconnects (language switch + failed send) must open exactly one socket.
import asyncio

import pipecat_gnani.stt as plugin_stt
from pipecat_gnani import GnaniSTTService

from bot import SafeGnaniSTTService


async def _race():
    opened = []

    async def fake_connect(self):
        await asyncio.sleep(0.05)  # handshake
        opened.append(1)
        self._ws = object()
        self._receive_task = asyncio.create_task(asyncio.sleep(10))

    real = plugin_stt.GnaniSTTService._connect
    plugin_stt.GnaniSTTService._connect = fake_connect
    try:
        stt = SafeGnaniSTTService(api_key="x", settings=GnaniSTTService.Settings())
        await asyncio.gather(stt._connect(), stt._connect())
        stt._receive_task.cancel()
    finally:
        plugin_stt.GnaniSTTService._connect = real
    return len(opened)


assert asyncio.run(_race()) == 1

with tempfile.TemporaryDirectory() as d:
    msgs = [{"role": "assistant", "content": "हाँ बेटा, कौन बोल रहा है?"}]
    assert json.loads(Path(save_call(msgs, Path(d))).read_text(encoding="utf-8")) == msgs

print("ok")
