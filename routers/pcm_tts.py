# File: routers/pcm_tts.py
# PCM Audio TTS API endpoint - returns raw PCM audio data

import logging
import numpy as np
import io
import os
import base64
import tempfile
import wave
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

import engine
from config import (
    get_gen_default_temperature,
    get_gen_default_exaggeration,
    get_gen_default_cfg_weight,
    get_gen_default_seed,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pcm-tts", tags=["PCM TTS"])


class PCMTTSRequest(BaseModel):
    """Request model for PCM TTS generation."""
    text: str = Field(..., description="Text to synthesize")
    sample_rate: int = Field(24000, description="Target sample rate (default: 24000 Hz)")
    temperature: Optional[float] = Field(None, ge=0.1, le=2.0, description="Sampling temperature (0.1-2.0)")
    exaggeration: Optional[float] = Field(None, ge=0.0, le=1.0, description="Emotion exaggeration level (0.0-1.0)")
    cfg_weight: Optional[float] = Field(None, ge=0.0, le=1.0, description="Classifier-free guidance weight (0.0-1.0)")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    reference_audio_base64: Optional[str] = Field(None, description="Base64-encoded WAV reference audio for voice cloning")


@router.post(
    "/generate",
    summary="Generate PCM audio",
    description="Generate text-to-speech audio and return raw PCM data (16-bit signed integer, little-endian)",
    responses={
        200: {
            "description": "Raw PCM audio data (16-bit signed integer, little-endian)",
            "content": {"audio/pcm": {}},
        },
        400: {
            "description": "Invalid parameters",
        },
        503: {
            "description": "TTS model not loaded",
        },
    },
)
async def generate_pcm_tts(request: PCMTTSRequest):
    """
    Generate TTS audio and return raw PCM data.
    
    **Parameters:**
    - **text**: Text to synthesize (required)
    - **sample_rate**: Target sample rate in Hz (default: 24000)
    - **temperature**: Controls randomness in generation (default: from config)
    - **exaggeration**: Controls expressiveness (default: from config)
    - **cfg_weight**: Classifier-free guidance weight (default: from config)
    - **seed**: Random seed for reproducibility (optional)
    
    **Returns:**
    - Raw PCM audio data (16-bit signed integer, little-endian, mono)
    - Content-Type: audio/pcm
    - Custom headers: X-Sample-Rate, X-Channels, X-Bit-Depth
    
    **Usage Example:**
    ```python
    import requests
    import numpy as np
    
    response = requests.post(
        "http://localhost:5000/pcm-tts/generate",
        json={"text": "Hello, world!", "sample_rate": 24000}
    )
    
    # Get metadata from headers
    sample_rate = int(response.headers['X-Sample-Rate'])
    
    # Convert PCM bytes to numpy array
    pcm_data = np.frombuffer(response.content, dtype=np.int16)
    ```
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
    
    try:
        # Get generation parameters (use defaults from config if not provided)
        temperature = request.temperature if request.temperature is not None else get_gen_default_temperature()
        exaggeration = request.exaggeration if request.exaggeration is not None else get_gen_default_exaggeration()
        cfg_weight = request.cfg_weight if request.cfg_weight is not None else get_gen_default_cfg_weight()
        seed = request.seed if request.seed is not None else get_gen_default_seed()
        
        logger.info(
            f"Generating PCM audio | Text: '{request.text[:50]}...' | "
            f"Target SR: {request.sample_rate}Hz | Temp: {temperature} | "
            f"Exag: {exaggeration} | CFG: {cfg_weight} | Seed: {seed}"
        )
        
        # Handle reference audio if provided
        audio_prompt_path = None
        temp_ref_path = None
        try:
            if request.reference_audio_base64:
                try:
                    audio_bytes = base64.b64decode(request.reference_audio_base64)
                    temp_ref = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    temp_ref.write(audio_bytes)
                    temp_ref.close()
                    temp_ref_path = temp_ref.name
                    audio_prompt_path = temp_ref_path
                    logger.info(f"Using reference audio ({len(audio_bytes)} bytes) for voice cloning")
                except Exception as ref_err:
                    logger.warning(f"Failed to decode reference audio, using default voice: {ref_err}")

            # Synthesize audio using engine
            audio_tensor, sr = engine.synthesize(
                text=request.text,
                audio_prompt_path=audio_prompt_path,
                temperature=temperature,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                seed=seed,
            )
        finally:
            # Clean up temp reference audio file
            if temp_ref_path and os.path.exists(temp_ref_path):
                try:
                    os.remove(temp_ref_path)
                except OSError:
                    pass
        
        if audio_tensor is None or sr is None:
            logger.error("TTS synthesis failed")
            raise HTTPException(
                status_code=500, 
                detail="TTS engine failed to synthesize audio."
            )
        
        # Convert tensor to numpy array
        audio_np = audio_tensor.cpu().numpy()
        
        # Ensure it's 1D (mono)
        if audio_np.ndim == 2:
            audio_np = audio_np.squeeze()
        
        # Resample if needed
        if request.sample_rate != sr:
            try:
                import librosa
                logger.info(f"Resampling from {sr}Hz to {request.sample_rate}Hz")
                audio_np = librosa.resample(
                    y=audio_np, 
                    orig_sr=sr, 
                    target_sr=request.sample_rate
                )
                sr = request.sample_rate
            except ImportError:
                logger.warning(
                    f"Librosa not available. Cannot resample. Using original rate {sr}Hz"
                )
            except Exception as e:
                logger.error(f"Resampling error: {e}", exc_info=True)
                # Continue with original sample rate
        
        # Clip audio to prevent overflow and convert to 16-bit PCM
        audio_clipped = np.clip(audio_np, -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767).astype(np.int16)
        
        # Convert to bytes (little-endian by default on most systems)
        pcm_bytes = audio_int16.tobytes()
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit = 2 bytes
            wav_file.setframerate(sr)
            wav_file.writeframes(pcm_bytes)
        
        wav_bytes = wav_buffer.getvalue()
        
        logger.info(
            f"Generated WAV audio: {len(wav_bytes)} bytes, "
            f"{len(audio_int16)} samples, {sr}Hz, "
            f"Duration: {len(audio_int16)/sr:.2f}s"
        )
        
        # Return WAV data (playable audio format with headers)
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "X-Sample-Rate": str(sr),
                "X-Channels": "1",  # Mono
                "X-Bit-Depth": "16",  # 16-bit
                "X-Duration": str(len(audio_int16) / sr),
                "Content-Length": str(len(wav_bytes)),
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in PCM TTS generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/info",
    summary="Get PCM TTS endpoint information",
    description="Returns information about the PCM TTS endpoint"
)
async def get_pcm_info():
    """Get information about the PCM TTS endpoint."""
    return {
        "endpoint": "/pcm-tts/generate",
        "method": "POST",
        "description": "Generate TTS audio and return raw PCM data",
        "format": {
            "encoding": "PCM",
            "bit_depth": 16,
            "byte_order": "little-endian",
            "channels": 1,
            "default_sample_rate": 24000,
            "supported_sample_rates": [8000, 16000, 22050, 24000, 44100, 48000]
        },
        "parameters": {
            "text": "Text to synthesize (required)",
            "sample_rate": "Target sample rate in Hz (default: 24000)",
            "temperature": "Sampling temperature 0.1-2.0 (optional)",
            "exaggeration": "Emotion exaggeration 0.0-1.0 (optional)",
            "cfg_weight": "CFG weight 0.0-1.0 (optional)",
            "seed": "Random seed for reproducibility (optional)"
        },
        "response_headers": {
            "X-Sample-Rate": "Actual sample rate of the audio",
            "X-Channels": "Number of audio channels (always 1)",
            "X-Bit-Depth": "Bit depth of the audio (always 16)",
            "X-Duration": "Duration in seconds",
            "Content-Length": "Size of PCM data in bytes"
        },
        "usage_example": {
            "python": """
import requests
import numpy as np

# Generate PCM audio
response = requests.post(
    "http://localhost:5000/pcm-tts/generate",
    json={
        "text": "Hello, world!",
        "sample_rate": 24000
    }
)

# Get metadata from headers
sample_rate = int(response.headers['X-Sample-Rate'])
duration = float(response.headers['X-Duration'])

# Convert PCM bytes to numpy array
pcm_data = np.frombuffer(response.content, dtype=np.int16)

# Convert to float [-1.0, 1.0] for processing
audio_float = pcm_data.astype(np.float32) / 32768.0
""",
            "curl": """
curl -X POST "http://localhost:5000/pcm-tts/generate" \\
  -H "Content-Type: application/json" \\
  -d '{"text": "Hello, world!", "sample_rate": 24000}' \\
  --output audio.pcm
"""
        }
    }
