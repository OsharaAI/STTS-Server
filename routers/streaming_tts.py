
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

        for audio_chunk, metrics in stream:
            if audio_chunk is None:
                continue
                
            # audio_chunk is a torch Tensor (1, samples)
            # Convert to numpy
            if isinstance(audio_chunk, torch.Tensor):
                audio_np = audio_chunk.squeeze().cpu().numpy()
            else:
                audio_np = audio_chunk
                
            # Convert to float32 bytes for transmission
            # We transmit raw float32 bytes. Client must know Sr and Format.
            # Usually Chatterbox is 24000Hz float32.
            
            # Ensure float32
            audio_np = audio_np.astype(np.float32)
            yield audio_np.tobytes()

    return StreamingResponse(
        audio_chunk_generator(),
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": str(24000), # Assuming default model SR
            "X-Encoding": "float32",
        }
    )
