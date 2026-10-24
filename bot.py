"""Ammamma: an AI grandmother who keeps scam callers busy.

Usage:
  python bot.py                                  # browser test at http://localhost:7860/client
  python bot.py -t twilio -x <name>.ngrok.app    # real phone number via Twilio
  python bot.py -t exotel -x <name>.ngrok.app    # real phone number via Exotel
"""

import asyncio
import contextlib
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    LLMMessagesAppendFrame,
    STTUpdateSettingsFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments, WebSocketRunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat_gnani import GnaniSTTService, GnaniTTSService

load_dotenv(override=True)

HERE = Path(__file__).parent
CALLS_DIR = HERE / "calls"

# Every language both Prisma STT and Timbre v2.5 support, with the voice Ammamma uses for it.
LANGUAGES = {
    "hi-IN": ("Hindi", "Chitra"),
    "en-IN": ("English", "Trupti"),  # "Indian English" made Gemini answer in Hindi
    "te-IN": ("Telugu", "Lehara"),
    "ta-IN": ("Tamil", "Trisha"),
    "kn-IN": ("Kannada", "Saanvi"),
    "ml-IN": ("Malayalam", "Reshma"),
    "mr-IN": ("Marathi", "Zahira"),
    "bn-IN": ("Bengali", "Kirra"),
    "gu-IN": ("Gujarati", "Falak"),
    "pa-IN": ("Punjabi", "Mehuli"),
}

SWITCH_LANGUAGE = FunctionSchema(
    name="switch_language",
    description=(
        "Switch the language you listen and speak in. Call it when the caller asks for "
        "another language or is clearly speaking one."
    ),
    properties={
        "code": {
            "type": "string",
            "enum": list(LANGUAGES),
            "description": ", ".join(f"{c} = {n}" for c, (n, _) in LANGUAGES.items()),
        }
    },
    required=["code"],
)

_telephony = lambda: FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True)
transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    "twilio": _telephony,
    "exotel": _telephony,
}


class SafeGnaniSTTService(GnaniSTTService):
    """Serializes the plugin's (re)connects.

    pipecat-gnani 0.5.12 reconnects from two places at once on a language switch: the
    settings update and run_stt's send-failure path. That leaves two receive loops on one
    socket ("cannot call recv while another coroutine is already running recv").
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._conn_lock = asyncio.Lock()

    async def _connect(self):
        async with self._conn_lock:
            if self._ws is not None:
                return  # someone else already reconnected
            if self._receive_task:  # run_stt drops _ws without stopping its reader
                self._receive_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._receive_task
                self._receive_task = None
            await super()._connect()

    async def _disconnect(self):
        async with self._conn_lock:
            await super()._disconnect()


def system_prompt(lang: str) -> str:
    persona = (HERE / "persona.md").read_text(encoding="utf-8")
    return persona.replace("{language}", LANGUAGES[lang][0])


def make_llm(env, prompt: str):
    """Self-hosted Evon (OpenAI-compatible) when EVON_BASE_URL is set, otherwise native Gemini.

    Gemini's OpenAI-compatible endpoint streams tool calls in a shape Pipecat splits into a
    phantom empty call, which breaks every request after a language switch.
    """
    if env.get("EVON_BASE_URL"):
        return OpenAILLMService(
            api_key=env.get("EVON_API_KEY", "none"),
            base_url=env["EVON_BASE_URL"],
            settings=OpenAILLMService.Settings(
                model=env.get("EVON_MODEL", "model"),
                system_instruction=prompt,
                temperature=0.6,
                top_p=0.95,
                # Evon's reasoning trace adds seconds of silence on a live call.
                extra={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
            ),
        )
    if not env.get("GEMINI_API_KEY"):
        raise RuntimeError("Set GEMINI_API_KEY, or EVON_BASE_URL for self-hosted Evon.")
    return GoogleLLMService(
        api_key=env["GEMINI_API_KEY"],
        settings=GoogleLLMService.Settings(
            # 2.5-flash with thinking off: ~1.2s to first token vs ~2.5s for 3.8-flash.
            model=env.get("GEMINI_MODEL", "gemini-2.5-flash"),
            system_instruction=prompt,
            thinking=GoogleLLMService.ThinkingConfig(thinking_budget=0),
        ),
    )


def save_call(messages: list, calls_dir: Path = CALLS_DIR) -> Path:
    calls_dir.mkdir(exist_ok=True)
    path = calls_dir / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    gnani_api_key = os.environ.get("GNANI_API_KEY")
    if not gnani_api_key:
        raise RuntimeError("GNANI_API_KEY is not set. Copy .env.example to .env and fill it in.")

    lang = os.environ.get("AMMAMMA_LANG", "hi-IN")
    if lang not in LANGUAGES:
        raise RuntimeError(f"AMMAMMA_LANG must be one of {', '.join(LANGUAGES)}, got {lang!r}")
    stt = SafeGnaniSTTService(
        api_key=gnani_api_key,
        settings=GnaniSTTService.Settings(language=Language(lang)),
    )
    tts = GnaniTTSService(
        api_key=gnani_api_key,
        settings=GnaniTTSService.Settings(
            voice=LANGUAGES[lang][1],
            model="timbre-v2.5",
            language=Language(lang),
        ),
    )
    llm = make_llm(os.environ, system_prompt(lang))

    async def switch_language(params):
        code = params.arguments.get("code")
        if code not in LANGUAGES:
            await params.result_callback({"error": f"Unsupported language {code!r}"})
            return
        name, voice = LANGUAGES[code]
        logger.info(f"Switching language to {name}")
        # TTS sits after the LLM: push in-line so the next reply is already in the new voice.
        await params.llm.push_frame(
            TTSUpdateSettingsFrame(delta=GnaniTTSService.Settings(voice=voice, language=Language(code)))
        )
        # STT sits before the LLM: enter from the pipeline source instead.
        await task.queue_frames(
            [STTUpdateSettingsFrame(delta=GnaniSTTService.Settings(language=Language(code)))]
        )
        await params.result_callback(
            {"now_speaking": name, "instruction": f"From now on reply only in {name}, in its native script."}
        )

    llm.register_function("switch_language", switch_language)

    context = LLMContext(tools=ToolsSchema(standard_tools=[SWITCH_LANGUAGE]))
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregators.user(),
            llm,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )

    # Phone lines are 8 kHz; keep the whole pipeline there so nothing resamples twice.
    rate = 8000 if isinstance(runner_args, WebSocketRunnerArguments) else 16000
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=rate,
            audio_out_sample_rate=rate,
            enable_metrics=True,
        ),
        # ponytail: fixed 2-minute idle cutoff; long silences are part of the act, raise if calls drop.
        idle_timeout_secs=120,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Call connected")
        await task.queue_frames(
            [
                LLMMessagesAppendFrame(
                    messages=[{"role": "user", "content": "(The phone rings. You pick up.)"}],
                    run_llm=True,
                )
            ]
        )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Call ended, transcript saved to {save_call(context.get_messages())}")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
