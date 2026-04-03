# File: routers/multilingual_tts.py
# Multilingual Chatterbox TTS API endpoints with voice cloning support

import logging
import uuid
import io
import tempfile
import os
from pathlib import Path
from typing import Optional

import torch
import torchaudio as ta
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse

from models import ErrorResponse
from config import get_output_path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/multilingual-tts", tags=["Multilingual TTS"])

# Supported languages for multilingual TTS
SUPPORTED_LANGUAGES = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tr": "Turkish",
    "zh": "Chinese",
}


def get_multilingual_model(request: Request):
    """Dependency to get multilingual TTS model from app state."""
    if not hasattr(request.app.state, 'multilingual_tts_model'):
        raise HTTPException(
            status_code=503, 
            detail="Multilingual TTS model not initialized"
        )
    return request.app.state.multilingual_tts_model


@router.get(
    "/languages",
    summary="Get supported languages",
    description="Returns a list of all supported languages for multilingual TTS"
)
async def get_supported_languages():
    """Get list of supported languages for multilingual TTS."""
    return {
        "count": len(SUPPORTED_LANGUAGES),
        "languages": [
            {"code": code, "name": name}
            for code, name in sorted(SUPPORTED_LANGUAGES.items())
        ]
    }


@router.post(
    "/generate",
    summary="Generate multilingual TTS audio",
    description="Generate speech in 24 languages with optional voice cloning",
    responses={
        200: {
            "description": "Successfully generated audio file (WAV format)",
            "content": {"audio/wav": {}},
        },
        400: {
            "model": ErrorResponse,
            "description": "Invalid parameters (unsupported language, missing text, etc.)",
        },
        503: {
            "model": ErrorResponse,
            "description": "Multilingual TTS model not loaded or unavailable",
        },
    },
)
async def generate_multilingual_tts(
    text: str = Form(..., description="Text to synthesize in the target language"),
    language: str = Form("en", description="Language code (e.g., 'en', 'fr', 'es', 'zh', 'ja')"),
    reference_audio: Optional[UploadFile] = File(None, description="Optional reference audio file for voice cloning"),
    exaggeration: float = Form(0.5, ge=0.0, le=1.0, description="Emotion exaggeration level (0.0-1.0)"),
    cfg_weight: float = Form(0.5, ge=0.0, le=1.0, description="Classifier-free guidance weight (0.0-1.0)"),
    temperature: float = Form(0.8, ge=0.1, le=2.0, description="Sampling temperature (0.1-2.0)"),
    repetition_penalty: float = Form(2.0, ge=1.0, le=3.0, description="Repetition penalty (1.0-3.0)"),
    min_p: float = Form(0.05, ge=0.0, le=1.0, description="Minimum probability threshold (0.0-1.0)"),
    top_p: float = Form(1.0, ge=0.0, le=1.0, description="Top-p (nucleus) sampling (0.0-1.0)"),
    seed: Optional[int] = Form(None, description="Random seed for reproducibility"),
    request: Request = None
):
    """
    Generate multilingual TTS audio with optional voice cloning.
    
    **Supported Languages (24):**
    ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, ne, nl, no, pl, pt, ru, sv, sw, tr, zh
    
    **Parameters:**
    - **text**: Text to synthesize (in target language)
    - **language**: Language code (default: 'en')
    - **reference_audio**: Optional audio file for voice cloning (WAV, MP3, FLAC)
    - **exaggeration**: Emotion intensity (0.0-1.0)
    - **cfg_weight**: Guidance strength (0.0-1.0)
    - **temperature**: Sampling temperature (0.1-2.0)
    - **repetition_penalty**: Penalty for repetitions (1.0-3.0)
    - **min_p**: Minimum probability (0.0-1.0)
    - **top_p**: Nucleus sampling threshold (0.0-1.0)
    - **seed**: Random seed for reproducibility
    
    **Returns:**
    - WAV audio file (streaming response)
    """
    
    # Get multilingual model
    try:
        multilingual_model = get_multilingual_model(request)
    except HTTPException:
        logger.error("Multilingual TTS model not available")
        raise HTTPException(
            status_code=503,
            detail="Multilingual TTS model not initialized. Please check server configuration."
        )
    
    # Validate language
    language = language.lower()
    if language not in SUPPORTED_LANGUAGES:
        supported_langs = ", ".join(sorted(SUPPORTED_LANGUAGES.keys()))
        logger.error(f"Unsupported language: {language}")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{language}'. Supported: {supported_langs}"
        )
    
    # Validate text
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        logger.info(f"Random seed set to: {seed}")
    
    # Handle reference audio if provided
    audio_prompt_path = None
    temp_file = None
    
    try:
        if reference_audio:
            # Read uploaded file
            audio_content = await reference_audio.read()
            
            if len(audio_content) == 0:
                raise HTTPException(status_code=400, detail="Reference audio file is empty")
            
            # Get file extension
            filename = reference_audio.filename or "audio.wav"
            file_ext = os.path.splitext(filename)[1].lower() or ".wav"
            
            # Validate audio format
            allowed_formats = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            if file_ext not in allowed_formats:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported audio format: {file_ext}. Supported: {', '.join(allowed_formats)}"
                )
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=file_ext,
                dir=str(get_output_path())
            )
            temp_file.write(audio_content)
            temp_file.close()
            audio_prompt_path = temp_file.name
            
            logger.info(f"Saved reference audio: {audio_prompt_path}")
        
        # Log generation info
        logger.info(
            f"Generating multilingual TTS | "
            f"Language: {language} ({SUPPORTED_LANGUAGES[language]}) | "
            f"Text: {text[:100]}{'...' if len(text) > 100 else ''} | "
            f"Ref Audio: {'Yes' if audio_prompt_path else 'No'} | "
            f"Temp: {temperature} | Exag: {exaggeration}"
        )
        
        # Generate audio
        wav = multilingual_model.generate(
            text=text,
            language_id=language,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p
        )
        
        # Convert to WAV format in memory
        audio_array = wav.squeeze(0).cpu().numpy()
        audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
        
        # Create in-memory buffer
        buffer = io.BytesIO()
        sample_rate = getattr(multilingual_model, 'sr', 24000)
        ta.save(buffer, audio_tensor, sample_rate, format="wav")
        buffer.seek(0)
        
        logger.info(f"Successfully generated multilingual TTS audio for language: {language}")
        
        # Generate unique filename
        output_filename = f"tts_multilingual_{language}_{uuid.uuid4().hex[:8]}.wav"
        
        # Return streaming response
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename={output_filename}"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating multilingual TTS: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating multilingual TTS audio: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if audio_prompt_path and os.path.exists(audio_prompt_path):
            try:
                os.remove(audio_prompt_path)
                logger.debug(f"Cleaned up temporary file: {audio_prompt_path}")
            except Exception as e:
                logger.warning(f"Could not delete temp file {audio_prompt_path}: {e}")


@router.get(
    "/info",
    summary="Get multilingual TTS information",
    description="Get information about the multilingual TTS model and its capabilities"
)
async def get_multilingual_info(request: Request):
    """Get information about the multilingual TTS model."""
    try:
        multilingual_model = get_multilingual_model(request)
        model_loaded = True
    except HTTPException:
        model_loaded = False
    
    return {
        "name": "Chatterbox Multilingual TTS",
        "version": "1.0.0",
        "model_loaded": model_loaded,
        "supported_languages": len(SUPPORTED_LANGUAGES),
        "languages": SUPPORTED_LANGUAGES,
        "features": [
            "23 language support",
            "Voice cloning via reference audio",
            "Adjustable emotion and prosody",
            "High-quality neural synthesis"
        ],
        "parameters": {
            "exaggeration": {"range": [0.0, 1.0], "default": 0.5},
            "temperature": {"range": [0.1, 2.0], "default": 0.8},
            "cfg_weight": {"range": [0.0, 1.0], "default": 0.5},
            "repetition_penalty": {"range": [1.0, 3.0], "default": 2.0},
        }
    }
