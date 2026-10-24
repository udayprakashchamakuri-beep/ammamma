"""Ammamma: an AI grandmother who keeps scam callers busy.

Usage:
  python bot.py                                  # browser test at http://localhost:7860/client
  python bot.py -t twilio -x <name>.ngrok.app    # real phone number via Twilio
  python bot.py -t exotel -x <name>.ngrok.app    # real phone number via Exotel
"""

import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesAppendFrame
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
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat_gnani import GnaniSTTService, GnaniTTSService

load_dotenv(override=True)

HERE = Path(__file__).parent
CALLS_DIR = HERE / "calls"

LANGUAGE_NAMES = {
    "hi-IN": "Hindi", "te-IN": "Telugu", "ta-IN": "Tamil", "kn-IN": "Kannada",
    "ml-IN": "Malayalam", "mr-IN": "Marathi", "bn-IN": "Bengali", "gu-IN": "Gujarati",
    "pa-IN": "Punjabi", "en-IN": "Indian English",
}

_telephony = lambda: FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True)
transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    "twilio": _telephony,
    "exotel": _telephony,
}


def system_prompt(lang: str) -> str:
    persona = (HERE / "persona.md").read_text(encoding="utf-8")
    return persona.replace("{language}", LANGUAGE_NAMES.get(lang, lang))


def make_llm(env, prompt: str):
    """Self-hosted Evon when EVON_BASE_URL is set, otherwise Groq."""
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
    if not env.get("GROQ_API_KEY"):
        raise RuntimeError("Set GROQ_API_KEY, or EVON_BASE_URL for self-hosted Evon.")
    return GroqLLMService(
        api_key=env["GROQ_API_KEY"],
        settings=GroqLLMService.Settings(
            model=env.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            system_instruction=prompt,
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
    stt = GnaniSTTService(
        api_key=gnani_api_key,
        settings=GnaniSTTService.Settings(language=Language(lang)),
    )
    tts = GnaniTTSService(
        api_key=gnani_api_key,
        settings=GnaniTTSService.Settings(
            voice=os.environ.get("AMMAMMA_VOICE", "Chitra"),
            model="timbre-v2.5",
            language=Language(lang),
        ),
    )
    llm = make_llm(os.environ, system_prompt(lang))

    context = LLMContext()
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
