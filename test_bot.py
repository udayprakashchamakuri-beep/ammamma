"""Run: .venv/Scripts/python test_bot.py"""

import json
import tempfile
from pathlib import Path

from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.openai.llm import OpenAILLMService

from bot import make_llm, save_call, system_prompt

prompt = system_prompt("te-IN")
assert "Telugu" in prompt and "{language}" not in prompt

assert isinstance(make_llm({"GROQ_API_KEY": "x"}, prompt), GroqLLMService)

evon = make_llm({"EVON_BASE_URL": "http://localhost:8000/v1", "GROQ_API_KEY": "x"}, prompt)
assert type(evon) is OpenAILLMService
assert evon._settings.extra["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False

try:
    make_llm({}, prompt)
    raise AssertionError("expected missing-key error")
except RuntimeError:
    pass

with tempfile.TemporaryDirectory() as d:
    msgs = [{"role": "assistant", "content": "हाँ बेटा, कौन बोल रहा है?"}]
    assert json.loads(Path(save_call(msgs, Path(d))).read_text(encoding="utf-8")) == msgs

print("ok")
