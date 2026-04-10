
# File: routers/streaming_tts.py
# Streaming TTS API endpoint

import logging
import os
import numpy as np
import shutil
import tempfile
import torch
import json
from typing import Optional, Generator
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import engine
from config import (
    get_gen_default_cfg_weight,
)
import utils

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Streaming TTS"])

DEFAULT_SAMPLE_RATE = 24000
# Smaller chunk size = lower time-to-first-audio for streaming.
# A moderately larger default reduces over-fragmented output and boundary hiccups.
DEFAULT_STREAM_CHUNK_SIZE = int(os.getenv("TTS_STREAM_DEFAULT_CHUNK_SIZE", "80"))
MIN_STREAM_CHUNK_SIZE = 10
MAX_STREAM_CHUNK_SIZE = 500


def _resolve_sample_rate() -> int:
    return int(getattr(engine.chatterbox_model, "sr", DEFAULT_SAMPLE_RATE))


def _normalize_chunk_size(raw_chunk_size: Optional[int]) -> int:
    try:
        chunk_size = int(raw_chunk_size) if raw_chunk_size is not None else DEFAULT_STREAM_CHUNK_SIZE
    except (TypeError, ValueError):
        return DEFAULT_STREAM_CHUNK_SIZE

    return max(MIN_STREAM_CHUNK_SIZE, min(MAX_STREAM_CHUNK_SIZE, chunk_size))


def _audio_chunk_generator(
    text: str,
    audio_prompt_path: Optional[str],
    temperature: float,
    exaggeration: float,
    cfg_weight: float,
    seed: int,
    chunk_size: int,
    language_id: Optional[str],
    repetition_penalty: float,
    min_p: float,
    top_p: float,
) -> Generator[bytes, None, None]:
    # Using engine.synthesize_stream
    stream = engine.synthesize_stream(
        text=text,
        audio_prompt_path=audio_prompt_path,
        temperature=temperature,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        seed=seed,
        chunk_size=chunk_size,
        language_id=language_id,
        repetition_penalty=repetition_penalty,
        min_p=min_p,
        top_p=top_p,
    )

    # Overlap-add crossfade to eliminate boundary artifacts between chunks.
    # Each chunk from the model has fade-in/fade-out applied (see _process_token_buffer).
    # We blend the overlapping fade regions to maintain smooth amplitude.
    sample_rate = _resolve_sample_rate()
    # Keep crossfade small for lower latency.
    crossfade_ms = int(os.getenv("TTS_STREAM_CROSSFADE_MS", "10"))  # ms
    crossfade_samples = int(crossfade_ms / 1000.0 * sample_rate)
    prev_tail = None  # holds the last crossfade_samples of the previous chunk

    # Trailing silence detection: stop streaming if model generates
    # consecutive silent chunks (hallucinated/garbage audio after speech ends)
    silence_threshold = 0.01  # RMS below this = silence
    consecutive_silent_chunks = 0
    max_silent_chunks = 3  # stop after 3 consecutive silent chunks
    has_produced_voiced = False  # track if any voiced audio was produced

    for audio_chunk, _metrics in stream:
        if audio_chunk is None:
            continue

        # audio_chunk is a torch Tensor (1, samples)
        if isinstance(audio_chunk, torch.Tensor):
            audio_np = audio_chunk.squeeze().cpu().numpy()
        else:
            audio_np = audio_chunk

        audio_np = audio_np.astype(np.float32)

        # Check if this chunk is mostly silence/noise
        chunk_rms = np.sqrt(np.mean(audio_np ** 2)) if len(audio_np) > 0 else 0.0
        if chunk_rms < silence_threshold:
            consecutive_silent_chunks += 1
            if has_produced_voiced and consecutive_silent_chunks >= max_silent_chunks:
                logger.info(
                    f"Stopping stream: {consecutive_silent_chunks} consecutive silent chunks "
                    f"detected after voiced audio (RMS={chunk_rms:.4f})"
                )
                break
        else:
            consecutive_silent_chunks = 0
            has_produced_voiced = True

        if prev_tail is not None and len(audio_np) > crossfade_samples and len(prev_tail) == crossfade_samples:
            # Overlap-add: blend previous chunk's faded-out tail with this chunk's faded-in head
            blended = prev_tail + audio_np[:crossfade_samples]
            # Yield the blended region + body (excluding the tail we'll hold back)
            body = audio_np[crossfade_samples:-crossfade_samples] if len(audio_np) > 2 * crossfade_samples else np.array([], dtype=np.float32)
            yield np.concatenate([blended, body]).tobytes()
            prev_tail = audio_np[-crossfade_samples:].copy()
        else:
            # First chunk or chunk too short for crossfade - hold back the tail
            if len(audio_np) > crossfade_samples:
                yield audio_np[:-crossfade_samples].tobytes()
                prev_tail = audio_np[-crossfade_samples:].copy()
            else:
                yield audio_np.tobytes()
                prev_tail = None

    # Flush the held-back tail of the last chunk with fade-out
    if prev_tail is not None:
        # Apply a gentle fade-out to the final tail for clean ending
        fade_samples = min(len(prev_tail), crossfade_samples)
        if fade_samples > 0:
            fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
            prev_tail[-fade_samples:] *= fade_out
        yield prev_tail.tobytes()


@router.post(
    "/stream",
    summary="Generate Streaming Audio",
    description="Generate text-to-speech audio and stream it as raw PCM chunks.",
)
async def generate_stream(
    background_tasks: BackgroundTasks,
    text: str = Form(..., description="Text to convert to speech"),
    reference_audio: Optional[UploadFile] = File(None, description="Optional reference audio file for voice cloning"),
    reference_audio_url: Optional[str] = Form(None, description="Optional reference audio URL for voice cloning (http/https/s3)"),
    exaggeration: float = Form(0.5, description="Voice exaggeration level (0.25-2.0)"),
    temperature: float = Form(0.8, description="Sampling temperature (0.05-5.0)"),
    cfg_weight: Optional[float] = Form(None, description="Classifier-free guidance weight (0.2-1.0)"),
    seed: int = Form(0, description="Random seed (0 for random)"),
    speed_factor: Optional[float] = Form(None, description="Global speech speed factor (0.25-4.0)"),
    diffusion_steps: int = Form(10, description="Number of diffusion steps (1-15)"),
    min_p: float = Form(0.05, description="Minimum probability sampler (0.0-1.0)"),
    top_p: float = Form(1.0, description="Top-p/nucleus sampling (0.0-1.0)"),
    repetition_penalty: float = Form(1.2, description="Repetition penalty (1.0-2.0)"),
    split_text: bool = Form(True, description="Whether to split text into chunks"),
    chunk_size: int = Form(
        DEFAULT_STREAM_CHUNK_SIZE,
        description="Speech token chunk size for streaming (lower = faster first chunk)",
        ge=MIN_STREAM_CHUNK_SIZE,
        le=MAX_STREAM_CHUNK_SIZE,
    ),
    language_id: Optional[str] = Form("ne", description="Language code for multilingual model (e.g., 'en', 'fr', 'es', 'zh')"),
):
    """
    Generate TTS audio and stream raw PCM data.
    """

    logger.info(
        f"Received streaming TTS request: text='{text[:30]}...', "
        f"reference_audio={'provided' if reference_audio and reference_audio.filename else 'none'}, "
        f"exaggeration={exaggeration}, temperature={temperature}, cfg_weight={cfg_weight}, "
        f"seed={seed}, speed_factor={speed_factor}, diffusion_steps={diffusion_steps}, "
        f"min_p={min_p}, top_p={top_p}, repetition_penalty={repetition_penalty}, "
        f"split_text={split_text}, chunk_size={chunk_size}, language_id={language_id}"
    )
    
    # Validate text
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Check if TTS model is loaded
    if not engine.MODEL_LOADED:
        logger.error("TTS model not loaded")
        raise HTTPException(
            status_code=503,
            detail="TTS engine model is not currently loaded or available."
        )
    if engine.has_cuda_device_assert():
        raise HTTPException(
            status_code=503,
            detail=(
                "CUDA device-side assert was triggered in this server process. "
                "Restart the container/process, then retry with CUDA_LAUNCH_BLOCKING=1 for accurate stack traces."
            ),
        )

    # Determine audio prompt path from optional reference upload.
    audio_prompt_path = None
    if reference_audio and reference_audio.filename:
        allowed_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
        file_ext = Path(reference_audio.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported reference audio format: {file_ext}. Supported: {', '.join(allowed_extensions)}",
            )

        temp_ref = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
        try:
            shutil.copyfileobj(reference_audio.file, temp_ref)
            temp_ref.close()
            audio_prompt_path = temp_ref.name
            background_tasks.add_task(lambda p: Path(p).unlink(missing_ok=True), temp_ref.name)
        except Exception as e:
            logger.error(f"Error processing reference audio: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Failed to process reference audio: {str(e)}")
        finally:
            await reference_audio.close()
    elif reference_audio_url:
        downloaded = utils.download_audio_from_url(reference_audio_url)
        if downloaded is None:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download reference_audio_url: {reference_audio_url}",
            )
        audio_prompt_path = str(downloaded)
        # Ensure temporary download is cleaned up after request finishes.
        background_tasks.add_task(
            lambda p: Path(p).unlink(missing_ok=True),
            audio_prompt_path,
        )

    # Keep parity with /generate for request schema while mapping only
    # stream-supported params to engine.synthesize_stream.
    if any([
        speed_factor is not None,
        diffusion_steps != 10,
        min_p != 0.05,
        top_p != 1.0,
        repetition_penalty != 1.2,
        split_text is not True,
    ]):
        logger.debug(
            "Stream endpoint received /generate-compatible params that are currently ignored by stream backend: "
            f"speed_factor={speed_factor}, diffusion_steps={diffusion_steps}, min_p={min_p}, "
            f"top_p={top_p}, repetition_penalty={repetition_penalty}, split_text={split_text}"
        )

    cfg_weight = cfg_weight if cfg_weight is not None else get_gen_default_cfg_weight()
    
    logger.info(
        f"Streaming for text: '{text[:30]}...' | "
        f"Voice: {'reference_audio' if audio_prompt_path else 'Default'} | "
        f"Lang: {language_id} | Seed: {seed}"
    )

    sample_rate = _resolve_sample_rate()

    return StreamingResponse(
        _audio_chunk_generator(
            text=text,
            audio_prompt_path=audio_prompt_path,
            temperature=temperature,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            seed=seed,
            chunk_size=chunk_size,
            language_id=language_id,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
        ),
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": str(sample_rate),
            "X-Encoding": "float32",
        }
    )


@router.websocket("/ws/stream")
async def websocket_stream_tts(websocket: WebSocket):
    """
    Stream TTS audio over WebSocket.

    Protocol:
    - Client sends JSON text message: {"action":"synthesize", "text":"...", ...optional params...}
    - Server sends JSON control messages and binary audio frames (float32 PCM @ 24kHz).
    """
    await websocket.accept()

    if not engine.MODEL_LOADED:
        await websocket.send_json({
            "type": "error",
            "message": "TTS engine model is not currently loaded or available.",
        })
        await websocket.close()
        return
    if engine.has_cuda_device_assert():
        await websocket.send_json({
            "type": "error",
            "message": (
                "CUDA device-side assert was triggered in this server process. "
                "Restart the container/process, then retry with CUDA_LAUNCH_BLOCKING=1 for accurate stack traces."
            ),
        })
        await websocket.close()
        return

    sample_rate = _resolve_sample_rate()

    logger.info("WebSocket streaming TTS connection established")

    await websocket.send_json({
        "type": "ready",
        "message": "Streaming TTS ready",
        "audio": {
            "sample_rate": sample_rate,
            "encoding": "float32",
        },
    })

    try:
        while True:
            data = await websocket.receive()

            if data["type"] == "websocket.disconnect":
                logger.info("WebSocket streaming TTS client disconnected")
                break

            if data["type"] != "websocket.receive" or "text" not in data:
                continue

            try:
                message = json.loads(data["text"])
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON command"})
                continue

            action = message.get("action")

            if action == "ping":
                await websocket.send_json({"type": "pong", "message": "WebSocket connection active"})
                continue

            if action != "synthesize":
                await websocket.send_json({
                    "type": "error",
                    "message": "Unsupported action. Use 'synthesize' or 'ping'.",
                })
                continue

            text = (message.get("text") or "").strip()
            if not text:
                await websocket.send_json({"type": "error", "message": "Text cannot be empty"})
                continue

            temperature = float(message.get("temperature", 0.8))
            exaggeration = float(message.get("exaggeration", 0.5))
            cfg_weight = float(message.get("cfg_weight", get_gen_default_cfg_weight()))
            seed = int(message.get("seed", 0))
            chunk_size = _normalize_chunk_size(message.get("chunk_size", DEFAULT_STREAM_CHUNK_SIZE))
            language_id = message.get("language_id") or None
            repetition_penalty = float(message.get("repetition_penalty", 1.2))
            min_p = float(message.get("min_p", 0.05))
            top_p = float(message.get("top_p", 1.0))

            # Optional reference audio URL for voice cloning.
            audio_prompt_path: Optional[str] = None
            ref_url = (message.get("reference_audio_url") or message.get("reference_audio") or "").strip()
            if ref_url:
                downloaded = utils.download_audio_from_url(ref_url)
                if downloaded is None:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": f"Failed to download reference_audio_url: {ref_url}",
                        }
                    )
                    continue
                audio_prompt_path = str(downloaded)

            await websocket.send_json({
                "type": "start",
                    "message": "Synthesis started",
                    "meta": {
                        "sample_rate": sample_rate,
                        "encoding": "float32",
                        "language_id": language_id,
                        "chunk_size": chunk_size,
                        "has_reference_audio": bool(audio_prompt_path),
                    },
                })

            try:
                chunk_count = 0
                try:
                    for chunk in _audio_chunk_generator(
                        text=text,
                        audio_prompt_path=audio_prompt_path,
                        temperature=temperature,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight,
                        seed=seed,
                        chunk_size=chunk_size,
                        language_id=language_id,
                        repetition_penalty=repetition_penalty,
                        min_p=min_p,
                        top_p=top_p,
                    ):
                        if not chunk:
                            continue
                        await websocket.send_bytes(chunk)
                        chunk_count += 1
                finally:
                    # Clean up any temporary reference audio file we downloaded.
                    if audio_prompt_path:
                        try:
                            Path(audio_prompt_path).unlink(missing_ok=True)
                        except Exception:
                            pass

                await websocket.send_json({
                    "type": "complete",
                    "message": "Synthesis complete",
                    "chunks": chunk_count,
                })
            except Exception as synthesis_error:
                logger.error(f"WebSocket streaming TTS synthesis error: {synthesis_error}", exc_info=True)
                await websocket.send_json({
                    "type": "error",
                    "message": f"Synthesis failed: {str(synthesis_error)}",
                })

    except WebSocketDisconnect:
        logger.info("WebSocket streaming TTS connection closed")
    except Exception as e:
        logger.error(f"WebSocket streaming TTS error: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Server error: {str(e)}",
            })
        except Exception:
            pass


class WarmupRequest(BaseModel):
    """Request model for TTS warmup."""
    voice_path: Optional[str] = Field(None, description="Path to voice file to pre-cache (uses default if not provided)")


@router.post(
    "/warmup",
    summary="Warmup TTS Model",
    description="Pre-cache voice conditioning and trigger T3 model compilation to eliminate first-request latency.",
)
async def warmup_tts(request: WarmupRequest = None):
    """
    Warmup the TTS model by pre-caching voice conditioning and running a dummy generation.
    Call this after server startup to eliminate cold-start latency on first real request.
    """
    if not engine.MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="TTS engine model is not currently loaded."
        )

    try:
        voice_path = request.voice_path if request else None
        engine.warmup_model(voice_path=voice_path)
        return {
            "status": "ok",
            "warmed_up": engine.is_warmed_up(),
            "message": "TTS model warmup complete. Voice conditioning cached and T3 compiled."
        }
    except Exception as e:
        logger.error(f"Warmup failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Warmup failed: {str(e)}")
