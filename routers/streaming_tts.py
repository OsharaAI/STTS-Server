
# File: routers/streaming_tts.py
# Streaming TTS API endpoint

import logging
import numpy as np
import io
import torch
from typing import Optional, Generator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import engine
from config import (
    get_gen_default_temperature,
    get_gen_default_exaggeration,
    get_gen_default_cfg_weight,
    get_gen_default_seed,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tts", tags=["Streaming TTS"])


class StreamingTTSRequest(BaseModel):
    """Request model for Streaming TTS generation."""
    text: str = Field(..., description="Text to synthesize")
    chunk_size: Optional[int] = Field(25, description="Number of tokens per chunk (default: 25)")
    temperature: Optional[float] = Field(None, ge=0.1, le=2.0, description="Sampling temperature (0.1-2.0)")
    exaggeration: Optional[float] = Field(None, ge=0.0, le=1.0, description="Emotion exaggeration level (0.0-1.0)")
    cfg_weight: Optional[float] = Field(None, ge=0.0, le=1.0, description="Classifier-free guidance weight (0.0-1.0)")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    language: Optional[str] = Field(None, description="Language code (e.g. 'en', 'fr')")
    predefined_voice_id: Optional[str] = Field(None, description="ID of predefined voice to use")
    reference_audio_filename: Optional[str] = Field(None, description="Filename of reference audio for cloning")


@router.post(
    "/stream",
    summary="Generate Streaming Audio",
    description="Generate text-to-speech audio and stream it as raw PCM chunks.",
)
async def generate_stream(request: StreamingTTSRequest):
    """
    Generate TTS audio and stream raw PCM data.
    """
    
    # Validate text
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Check if TTS model is loaded
    if not engine.MODEL_LOADED:
        logger.error("TTS model not loaded")
        raise HTTPException(
            status_code=503,
            detail="TTS engine model is not currently loaded or available."
        )

    # Determine audio prompt path
    audio_prompt_path = None
    if request.predefined_voice_id:
        from config import get_predefined_voices_path
        potential_path = get_predefined_voices_path(ensure_absolute=True) / request.predefined_voice_id
        if potential_path.is_file():
            audio_prompt_path = str(potential_path)
            
    elif request.reference_audio_filename:
        from config import get_reference_audio_path
        potential_path = get_reference_audio_path(ensure_absolute=True) / request.reference_audio_filename
        if potential_path.is_file():
            audio_prompt_path = str(potential_path)

    # Get generation parameters
    temperature = request.temperature if request.temperature is not None else get_gen_default_temperature()
    exaggeration = request.exaggeration if request.exaggeration is not None else get_gen_default_exaggeration()
    cfg_weight = request.cfg_weight if request.cfg_weight is not None else get_gen_default_cfg_weight()
    seed = request.seed if request.seed is not None else get_gen_default_seed()
    
    logger.info(
        f"Streaming for text: '{request.text[:30]}...' | "
        f"Voice: {request.predefined_voice_id or request.reference_audio_filename or 'Default'} | "
        f"Lang: {request.language} | Seed: {seed}"
    )

    def audio_chunk_generator() -> Generator[bytes, None, None]:
        # Using engine.synthesize_stream
        stream = engine.synthesize_stream(
            text=request.text,
            audio_prompt_path=audio_prompt_path,
            temperature=temperature,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            seed=seed,
            chunk_size=request.chunk_size or 25,
            language_id=request.language
        )

        # Overlap-add crossfade to eliminate boundary artifacts between chunks.
        # Each chunk from the model has fade-in/fade-out applied (see _process_token_buffer).
        # We blend the overlapping fade regions to maintain smooth amplitude.
        sample_rate = 24000  # Chatterbox default
        crossfade_ms = 20  # must match fade_duration in _process_token_buffer (0.02s)
        crossfade_samples = int(crossfade_ms / 1000.0 * sample_rate)
        prev_tail = None  # holds the last crossfade_samples of the previous chunk

        for audio_chunk, metrics in stream:
            if audio_chunk is None:
                continue
                
            # audio_chunk is a torch Tensor (1, samples)
            if isinstance(audio_chunk, torch.Tensor):
                audio_np = audio_chunk.squeeze().cpu().numpy()
            else:
                audio_np = audio_chunk
                
            audio_np = audio_np.astype(np.float32)

            if prev_tail is not None and len(audio_np) > crossfade_samples and len(prev_tail) == crossfade_samples:
                # Overlap-add: blend previous chunk's faded-out tail with this chunk's faded-in head
                blended = prev_tail + audio_np[:crossfade_samples]
                # Yield the blended region + body (excluding the tail we'll hold back)
                body = audio_np[crossfade_samples:-crossfade_samples] if len(audio_np) > 2 * crossfade_samples else np.array([], dtype=np.float32)
                yield np.concatenate([blended, body]).tobytes()
                prev_tail = audio_np[-crossfade_samples:].copy()
            else:
                # First chunk or chunk too short for crossfade — hold back the tail
                if len(audio_np) > crossfade_samples:
                    yield audio_np[:-crossfade_samples].tobytes()
                    prev_tail = audio_np[-crossfade_samples:].copy()
                else:
                    yield audio_np.tobytes()
                    prev_tail = None

        # Flush the held-back tail of the last chunk
        if prev_tail is not None:
            yield prev_tail.tobytes()

    return StreamingResponse(
        audio_chunk_generator(),
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": str(24000), # Assuming default model SR
            "X-Encoding": "float32",
        }
    )


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
