# File: server.py
# Main FastAPI application for the TTS Server.
# Handles API requests for text-to-speech generation, UI serving,
# configuration management, and file uploads.

import os
import io
import gc
import inspect
import logging
import logging.handlers  # For RotatingFileHandler
import re
import shutil
import time
import uuid
import yaml  # For loading presets
import numpy as np
import torch  # For device detection in multilingual TTS
import librosa  # For potential direct use if needed, though utils.py handles most
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any, Literal
import webbrowser  # For automatic browser opening
import threading  # For automatic browser opening

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    File,
    UploadFile,
    Form,
    BackgroundTasks,
    Depends,
)
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
    FileResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# --- Internal Project Imports ---
from config import (
    config_manager,
    get_host,
    get_port,
    get_log_file_path,
    get_output_path,
    get_reference_audio_path,
    get_predefined_voices_path,
    get_ui_title,
    get_gen_default_temperature,
    get_gen_default_exaggeration,
    get_gen_default_cfg_weight,
    get_gen_default_seed,
    get_gen_default_speed_factor,
    get_gen_default_language,
    get_audio_sample_rate,
    get_full_config_for_template,
    get_audio_output_format,
)

import engine  # TTS Engine interface
from stt_engine import STTEngine  # STT Engine class
from models import (  # Pydantic models
    CustomTTSRequest,
    ErrorResponse,
    UpdateStatusResponse,
    STTResponse,
)
import utils  # Utility functions

from pydantic import BaseModel, Field

# Import routers
from routers import stt, conversation
from routers.websocket import websocket_stt, websocket_conversation, websocket_conversation_v2
from routers.stt import get_stt_engine
from routers import multilingual_tts
from routers import pcm_tts
from routers import streaming_tts
from routers import chatterbox_streaming

try:
    from langdetect import detect
    from langdetect.lang_detect_exception import LangDetectException
except ImportError:
    detect = None
    LangDetectException = Exception

try:
    import nepali_num2word  # type: ignore
except ImportError:
    nepali_num2word = None


class OpenAISpeechRequest(BaseModel):
    model: str
    input_: str = Field(..., alias="input")
    voice: str
    response_format: Literal["wav", "opus", "mp3"] = "wav"  # Add "mp3"
    speed: float = 1.0
    seed: Optional[int] = None


# --- Logging Configuration ---
log_file_path_obj = get_log_file_path()
log_file_max_size_mb = config_manager.get_int("server.log_file_max_size_mb", 10)
log_backup_count = config_manager.get_int("server.log_file_backup_count", 5)

log_file_path_obj.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.handlers.RotatingFileHandler(
            str(log_file_path_obj),
            maxBytes=log_file_max_size_mb * 1024 * 1024,
            backupCount=log_backup_count,
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("watchfiles").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- Global Variables & Application Setup ---
startup_complete_event = threading.Event()  # For coordinating browser opening

_DEVANAGARI_TO_ASCII_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_NEPALI_DIGIT_PATTERN = re.compile(r"[0-9०-९]+")


def _resolve_nepali_num2word_fn():
    """Resolve a callable from nepali-num2word across possible API names."""
    if nepali_num2word is None:
        return None

    candidates = [
        "num2word",
        "num_to_word",
        "number_to_word",
        "convert_num_to_word",
        "convert_number_to_word",
        "convert_to_words",
        "convert",
    ]
    for name in candidates:
        fn = getattr(nepali_num2word, name, None)
        if callable(fn):
            return fn
    return None


_NEPALI_NUM2WORD_FN = _resolve_nepali_num2word_fn()
_NEPALI_NUM2WORD_SUPPORTS_LANG = False
if _NEPALI_NUM2WORD_FN is not None:
    try:
        _NEPALI_NUM2WORD_SUPPORTS_LANG = "lang" in inspect.signature(_NEPALI_NUM2WORD_FN).parameters
    except Exception:
        _NEPALI_NUM2WORD_SUPPORTS_LANG = False


def _should_apply_nepali_num2word(text: str, language_id: Optional[str]) -> bool:
    """Enable Nepali number normalization for explicit or detected Nepali text."""
    lang = (language_id or "").strip().lower()
    if lang in {"ne", "nepali"}:
        return True

    if detect is None or not text or not text.strip():
        return False

    try:
        return detect(text) == "ne"
    except LangDetectException:
        return False
    except Exception:
        return False


def _convert_nepali_numbers_to_words(text: str) -> str:
    """Convert ASCII/Devanagari integer tokens to Nepali words when possible."""
    if not text or _NEPALI_NUM2WORD_FN is None:
        return text

    def _replace(match: re.Match) -> str:
        token = match.group(0)
        ascii_token = token.translate(_DEVANAGARI_TO_ASCII_DIGITS)
        if not ascii_token.isdigit():
            return token

        try:
            # Try int first (common API), then fallback to raw string.
            if _NEPALI_NUM2WORD_SUPPORTS_LANG:
                converted = _NEPALI_NUM2WORD_FN(int(ascii_token), lang="np")
            else:
                converted = _NEPALI_NUM2WORD_FN(int(ascii_token))
        except Exception:
            try:
                if _NEPALI_NUM2WORD_SUPPORTS_LANG:
                    converted = _NEPALI_NUM2WORD_FN(ascii_token, lang="np")
                else:
                    converted = _NEPALI_NUM2WORD_FN(ascii_token)
            except Exception:
                return token

        return str(converted) if converted is not None else token

    return _NEPALI_DIGIT_PATTERN.sub(_replace, text)


def _log_num2word_conversion(endpoint_name: str, original_text: str, normalized_text: str) -> None:
    """Log original and normalized text when Nepali num2word changes input."""
    if normalized_text == original_text:
        return

    before_preview = original_text[:250].replace("\n", " ")
    after_preview = normalized_text[:250].replace("\n", " ")
    logger.info(f"[{endpoint_name}] Nepali num2word applied")
    logger.info(f"[{endpoint_name}] text_before: {before_preview}{'...' if len(original_text) > 250 else ''}")
    logger.info(f"[{endpoint_name}] text_after: {after_preview}{'...' if len(normalized_text) > 250 else ''}")


def _clear_cuda_memory_after_generate() -> None:
    """Release cached CUDA memory after /generate requests to reduce OOM risk."""
    gc.collect()
    if not torch.cuda.is_available():
        return

    if engine.has_cuda_device_assert():
        logger.warning(
            "Skipping CUDA cache clear because CUDA device-side assert was previously triggered in this process."
        )
        return

    try:
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        logger.debug("CUDA cache cleared after /generate request")
    except Exception as e:
        logger.warning(f"Failed to clear CUDA cache after /generate: {e}")


def _split_chunk_for_retry(chunk: str, min_chunk_size: int = 50) -> List[str]:
    """Split an oversized/failed chunk into smaller pieces for retry."""
    if not chunk or not chunk.strip():
        return []

    target = max(min_chunk_size, min(220, max(min_chunk_size, len(chunk) // 2)))
    sub_chunks = utils.chunk_text_by_sentences(chunk, target)

    # Fallback for very long single-sentence text where sentence splitter cannot split.
    if len(sub_chunks) <= 1 and len(chunk) > target:
        words = chunk.split()
        if len(words) > 1:
            mid = len(words) // 2
            left = " ".join(words[:mid]).strip()
            right = " ".join(words[mid:]).strip()
            sub_chunks = [c for c in [left, right] if c]

    # Safety: avoid returning the same unsplittable chunk.
    if len(sub_chunks) == 1 and sub_chunks[0].strip() == chunk.strip():
        return []

    return [c for c in sub_chunks if c and c.strip()]


def _cuda_assert_http_detail() -> str:
    """Build a consistent, actionable API error for CUDA assert failures."""
    last_error = engine.get_cuda_device_assert_details()
    suffix = f" Last error: {last_error}" if last_error else ""
    return (
        "CUDA device-side assert was triggered in this server process. "
        "The CUDA context is now invalid for further synthesis. "
        "Restart the container/process, then retry with CUDA_LAUNCH_BLOCKING=1 for accurate stack traces."
        f"{suffix}"
    )


def _enforce_max_chunk_size(chunks: List[str], max_chars: int) -> List[str]:
    """Hard-wrap oversized chunks so each synthesis call stays within safe length."""
    if max_chars <= 0:
        return [c for c in chunks if c and c.strip()]

    out: List[str] = []
    for chunk in chunks:
        if not chunk or not chunk.strip():
            continue
        if len(chunk) <= max_chars:
            out.append(chunk.strip())
            continue

        # Prefer word-boundary splitting for natural prosody.
        words = chunk.split()
        if len(words) > 1:
            current: List[str] = []
            current_len = 0
            for w in words:
                w_len = len(w)
                proposed = current_len + (1 if current else 0) + w_len
                if current and proposed > max_chars:
                    out.append(" ".join(current))
                    current = [w]
                    current_len = w_len
                else:
                    current.append(w)
                    current_len = proposed
            if current:
                out.append(" ".join(current))
            continue

        # Fallback for scripts/inputs without spaces.
        text = chunk.strip()
        for i in range(0, len(text), max_chars):
            out.append(text[i:i + max_chars])

    return [c for c in out if c and c.strip()]


# --- Dependency Functions ---
# STT engine dependency moved to routers


def _delayed_browser_open(host: str, port: int):
    """
    Waits for the startup_complete_event, then opens the web browser
    to the server's main page after a short delay.
    """
    try:
        startup_complete_event.wait(timeout=30)
        if not startup_complete_event.is_set():
            logger.warning(
                "Server startup did not signal completion within timeout. Browser will not be opened automatically."
            )
            return

        time.sleep(1.5)
        display_host = "localhost" if host == "0.0.0.0" else host
        browser_url = f"http://{display_host}:{port}/"
        logger.info(f"Attempting to open web browser to: {browser_url}")
        webbrowser.open(browser_url)
    except Exception as e:
        logger.error(f"Failed to open browser automatically: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application startup and shutdown events."""
    logger.info("TTS Server: Initializing application...")
    try:
        logger.info(f"Configuration loaded. Log file at: {get_log_file_path()}")

        paths_to_ensure = [
            get_output_path(),
            get_reference_audio_path(),
            get_predefined_voices_path(),
            Path("ui"),
            config_manager.get_path(
                "paths.model_cache", "./model_cache", ensure_absolute=True
            ),
        ]
        for p in paths_to_ensure:
            p.mkdir(parents=True, exist_ok=True)

        # Load TTS model
        tts_loaded = engine.load_model()
        if not tts_loaded:
            logger.critical(
                "CRITICAL: TTS Model failed to load on startup. TTS functionality will not work."
            )
        else:
            logger.info("TTS Model loaded successfully.")
            # Warmup: pre-cache voice conditioning and trigger T3 model compilation
            # This eliminates the ~2-5s latency hit on the first TTS request
            try:
                logger.info("Running TTS model warmup (voice conditioning + T3 compilation)...")
                engine.warmup_model()
                logger.info("TTS model warmup complete.")
            except Exception as e_warmup:
                logger.warning(f"TTS warmup failed (non-fatal, first request will be slower): {e_warmup}")
        
        # Initialize and load STT engine
        app.state.stt_engine = STTEngine()
        stt_loaded = app.state.stt_engine.load_model()
        if not stt_loaded:
            logger.warning(
                "WARNING: STT Model failed to load on startup. STT functionality will not work."
            )
        else:
            logger.info("STT Model loaded successfully.")
        
        # Initialize multilingual TTS endpoint model.
        # Prefer reusing engine's already-loaded model to avoid duplicating large GPU allocations.
        logger.info("Initializing multilingual TTS endpoint model...")
        app.state.multilingual_tts_model = None
        try:
            if engine.MODEL_LOADED and engine.chatterbox_model is not None:
                generate_sig = inspect.signature(engine.chatterbox_model.generate)
                if "language_id" in generate_sig.parameters:
                    app.state.multilingual_tts_model = engine.chatterbox_model
                    logger.info("Reusing engine multilingual model for /multilingual-tts endpoints (no duplicate GPU model load).")
                else:
                    logger.info("Engine model is standard TTS; multilingual endpoint model will be loaded separately.")

            if app.state.multilingual_tts_model is None:
                logger.info("Importing ChatterboxMultilingualTTS from chatterbox.mtl_tts...")
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                logger.info("Import successful. Loading dedicated Chatterbox Multilingual TTS model...")

                if torch.cuda.is_available():
                    mtl_device = "cuda"
                elif torch.backends.mps.is_available():
                    mtl_device = "mps"
                else:
                    mtl_device = "cpu"

                logger.info(f"Initializing multilingual TTS model on device: {mtl_device}")
                app.state.multilingual_tts_model = ChatterboxMultilingualTTS.from_pretrained(device=mtl_device)
                logger.info(f"Multilingual TTS model loaded successfully on {mtl_device}")
        except ImportError as ie:
            logger.warning(f"Failed to import Multilingual TTS module: {ie}. Multilingual TTS endpoints will not work.")
            app.state.multilingual_tts_model = None
        except Exception as e:
            logger.warning(f"Failed to initialize multilingual TTS model: {e}. Multilingual TTS endpoints will not work.", exc_info=True)
            app.state.multilingual_tts_model = None
        
        if tts_loaded:  # Only open browser if TTS (primary functionality) is working
            host_address = get_host()
            server_port = get_port()
            browser_thread = threading.Thread(
                target=lambda: _delayed_browser_open(host_address, server_port),
                daemon=True,
            )
            browser_thread.start()

        logger.info("Application startup sequence complete.")
        startup_complete_event.set()
        yield
    except Exception as e_startup:
        logger.error(
            f"FATAL ERROR during application startup: {e_startup}", exc_info=True
        )
        startup_complete_event.set()
        yield
    finally:
        logger.info("TTS Server: Application shutdown sequence initiated...")
        logger.info("TTS Server: Application shutdown complete.")


# --- FastAPI Application Instance ---
app = FastAPI(
    title=get_ui_title(),
    description="Text-to-Speech server with advanced UI and API capabilities.",
    version="2.0.2",  # Version Bump
    lifespan=lifespan,
    root_path="/tts",
)

from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def clear_cuda_cache_for_generate_requests(request: Request, call_next):
    """Ensure CUDA cache is released after each /generate request."""
    try:
        return await call_next(request)
    finally:
        if request.method == "POST" and request.url.path.endswith("/generate"):
            _clear_cuda_memory_after_generate()

# --- Include Routers ---
app.include_router(stt.router)
app.include_router(conversation.router)
app.include_router(websocket_stt.router)
app.include_router(websocket_conversation.router)
app.include_router(websocket_conversation_v2.router)  # New modular conversation library
app.include_router(multilingual_tts.router)  # Multilingual TTS
app.include_router(pcm_tts.router)  # PCM TTS
app.include_router(streaming_tts.router)  # Streaming TTS
app.include_router(chatterbox_streaming.router)  # Chatterbox voice-clone streaming

# --- Static Files and HTML Templates ---
ui_static_path = Path(__file__).parent / "ui"
if ui_static_path.is_dir():
    app.mount("/ui", StaticFiles(directory=ui_static_path), name="ui_static_assets")
else:
    logger.warning(
        f"UI static assets directory not found at '{ui_static_path}'. UI may not load correctly."
    )

# This will serve files from 'ui_static_path/vendor' when requests come to '/vendor/*'
if (ui_static_path / "vendor").is_dir():
    app.mount(
        "/vendor", StaticFiles(directory=ui_static_path / "vendor"), name="vendor_files"
    )
else:
    logger.warning(
        f"Vendor directory not found at '{ui_static_path}' /vendor. Wavesurfer might not load."
    )


@app.get("/styles.css", include_in_schema=False)
async def get_main_styles():
    styles_file = ui_static_path / "styles.css"
    if styles_file.is_file():
        return FileResponse(styles_file)
    raise HTTPException(status_code=404, detail="styles.css not found")


@app.get("/script.js", include_in_schema=False)
async def get_main_script():
    script_file = ui_static_path / "script.js"
    if script_file.is_file():
        return FileResponse(script_file)
    raise HTTPException(status_code=404, detail="script.js not found")


outputs_static_path = get_output_path(ensure_absolute=True)
try:
    app.mount(
        "/outputs",
        StaticFiles(directory=str(outputs_static_path)),
        name="generated_outputs",
    )
except RuntimeError as e_mount_outputs:
    logger.error(
        f"Failed to mount /outputs directory '{outputs_static_path}': {e_mount_outputs}. "
        "Output files may not be accessible via URL."
    )

templates = Jinja2Templates(directory=str(ui_static_path))

# --- API Endpoints ---


# --- Main UI Route ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def get_web_ui(request: Request):
    """Serves the main web interface (index.html)."""
    logger.info("Request received for main UI page ('/').")
    try:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"request": request},
        )
    except Exception as e_render:
        logger.error(f"Error rendering main UI page: {e_render}", exc_info=True)
        return HTMLResponse(
            "<html><body><h1>Internal Server Error</h1><p>Could not load the TTS interface. "
            "Please check server logs for more details.</p></body></html>",
            status_code=500,
        )


# --- API Endpoint for Initial UI Data ---
@app.get("/api/ui/initial-data", tags=["UI Helpers"])
async def get_ui_initial_data():
    """
    Provides all necessary initial data for the UI to render,
    including configuration, file lists, and presets.
    """
    logger.info("Request received for /api/ui/initial-data.")
    try:
        full_config = get_full_config_for_template()
        reference_files = utils.get_valid_reference_files()
        predefined_voices = utils.get_predefined_voices()
        loaded_presets = []
        presets_file = ui_static_path / "presets.yaml"
        if presets_file.exists():
            with open(presets_file, "r", encoding="utf-8") as f:
                yaml_content = yaml.safe_load(f)
                if isinstance(yaml_content, list):
                    loaded_presets = yaml_content
                else:
                    logger.warning(
                        f"Invalid format in {presets_file}. Expected a list, got {type(yaml_content)}."
                    )
        else:
            logger.info(
                f"Presets file not found: {presets_file}. No presets will be loaded for initial data."
            )

        initial_gen_result_placeholder = {
            "outputUrl": None,
            "filename": None,
            "genTime": None,
            "submittedVoiceMode": None,
            "submittedPredefinedVoice": None,
            "submittedCloneFile": None,
        }

        return {
            "config": full_config,
            "reference_files": reference_files,
            "predefined_voices": predefined_voices,
            "presets": loaded_presets,
            "initial_gen_result": initial_gen_result_placeholder,
        }
    except Exception as e:
        logger.error(f"Error preparing initial UI data for API: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to load initial data for UI."
        )


# --- Configuration Management API Endpoints ---
@app.post("/save_settings", response_model=UpdateStatusResponse, tags=["Configuration"])
async def save_settings_endpoint(request: Request):
    """
    Saves partial configuration updates to the config.yaml file.
    Merges the update with the current configuration.
    """
    logger.info("Request received for /save_settings.")
    try:
        partial_update = await request.json()
        if not isinstance(partial_update, dict):
            raise ValueError("Request body must be a JSON object for /save_settings.")
        logger.debug(f"Received partial config data to save: {partial_update}")

        if config_manager.update_and_save(partial_update):
            restart_needed = any(
                key in partial_update
                for key in ["server", "tts_engine", "paths", "model"]
            )
            message = "Settings saved successfully."
            if restart_needed:
                message += " A server restart may be required for some changes to take full effect."
            return UpdateStatusResponse(message=message, restart_needed=restart_needed)
        else:
            logger.error(
                "Failed to save configuration via config_manager.update_and_save."
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to save configuration file due to an internal error.",
            )
    except ValueError as ve:
        logger.error(f"Invalid data format for /save_settings: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid request data: {str(ve)}")
    except Exception as e:
        logger.error(f"Error processing /save_settings request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during settings save: {str(e)}",
        )


@app.post(
    "/reset_settings", response_model=UpdateStatusResponse, tags=["Configuration"]
)
async def reset_settings_endpoint():
    """Resets the configuration in config.yaml back to hardcoded defaults."""
    logger.warning("Request received to reset all configurations to default values.")
    try:
        if config_manager.reset_and_save():
            logger.info("Configuration successfully reset to defaults and saved.")
            return UpdateStatusResponse(
                message="Configuration reset to defaults. Please reload the page. A server restart may be beneficial.",
                restart_needed=True,
            )
        else:
            logger.error("Failed to reset and save configuration via config_manager.")
            raise HTTPException(
                status_code=500, detail="Failed to reset and save configuration file."
            )
    except Exception as e:
        logger.error(f"Error processing /reset_settings request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during settings reset: {str(e)}",
        )


@app.post(
    "/restart_server", response_model=UpdateStatusResponse, tags=["Configuration"]
)
async def restart_server_endpoint():
    """Attempts to trigger a server restart."""
    logger.info("Request received for /restart_server.")
    message = (
        "Server restart initiated. If running locally without a process manager, "
        "you may need to restart manually. For managed environments (Docker, systemd), "
        "the manager should handle the restart."
    )
    logger.warning(message)
    return UpdateStatusResponse(message=message, restart_needed=True)


# --- UI Helper API Endpoints ---
@app.get("/get_reference_files", response_model=List[str], tags=["UI Helpers"])
async def get_reference_files_api():
    """Returns a list of valid reference audio filenames (.wav, .mp3)."""
    logger.debug("Request for /get_reference_files.")
    try:
        return utils.get_valid_reference_files()
    except Exception as e:
        logger.error(f"Error getting reference files for API: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve reference audio files."
        )


@app.get(
    "/get_predefined_voices", response_model=List[Dict[str, str]], tags=["UI Helpers"]
)
async def get_predefined_voices_api():
    """Returns a list of predefined voices with display names and filenames."""
    logger.debug("Request for /get_predefined_voices.")
    try:
        return utils.get_predefined_voices()
    except Exception as e:
        logger.error(f"Error getting predefined voices for API: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve predefined voices list."
        )


# --- File Upload Endpoints ---
@app.post("/upload_reference", tags=["File Management"])
async def upload_reference_audio_endpoint(files: List[UploadFile] = File(...)):
    """
    Handles uploading of reference audio files (.wav, .mp3) for voice cloning.
    Validates files and saves them to the configured reference audio path.
    """
    logger.info(f"Request to /upload_reference with {len(files)} file(s).")
    ref_path = get_reference_audio_path(ensure_absolute=True)
    uploaded_filenames_successfully: List[str] = []
    upload_errors: List[Dict[str, str]] = []

    for file in files:
        if not file.filename:
            upload_errors.append(
                {"filename": "Unknown", "error": "File received with no filename."}
            )
            logger.warning("Upload attempt with no filename.")
            continue

        safe_filename = utils.sanitize_filename(file.filename)
        destination_path = ref_path / safe_filename

        try:
            if not (
                safe_filename.lower().endswith(".wav")
                or safe_filename.lower().endswith(".mp3")
            ):
                raise ValueError("Invalid file type. Only .wav and .mp3 are allowed.")

            if destination_path.exists():
                logger.info(
                    f"Reference file '{safe_filename}' already exists. Skipping duplicate upload."
                )
                if safe_filename not in uploaded_filenames_successfully:
                    uploaded_filenames_successfully.append(safe_filename)
                continue

            with open(destination_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(
                f"Successfully saved uploaded reference file to: {destination_path}"
            )

            max_duration = config_manager.get_int(
                "audio_output.max_reference_duration_sec", 30
            )
            is_valid, validation_msg = utils.validate_reference_audio(
                destination_path, max_duration
            )
            if not is_valid:
                logger.warning(
                    f"Uploaded file '{safe_filename}' failed validation: {validation_msg}. Deleting."
                )
                destination_path.unlink(missing_ok=True)
                upload_errors.append(
                    {"filename": safe_filename, "error": validation_msg}
                )
            else:
                uploaded_filenames_successfully.append(safe_filename)

        except Exception as e_upload:
            error_msg = f"Error processing file '{file.filename}': {str(e_upload)}"
            logger.error(error_msg, exc_info=True)
            upload_errors.append({"filename": file.filename, "error": str(e_upload)})
        finally:
            await file.close()

    all_current_reference_files = utils.get_valid_reference_files()
    response_data = {
        "message": f"Processed {len(files)} file(s).",
        "uploaded_files": uploaded_filenames_successfully,
        "all_reference_files": all_current_reference_files,
        "errors": upload_errors,
    }
    status_code = (
        200 if not upload_errors or len(uploaded_filenames_successfully) > 0 else 400
    )
    if upload_errors:
        logger.warning(
            f"Upload to /upload_reference completed with {len(upload_errors)} error(s)."
        )
    return JSONResponse(content=response_data, status_code=status_code)


@app.post("/upload_predefined_voice", tags=["File Management"])
async def upload_predefined_voice_endpoint(files: List[UploadFile] = File(...)):
    """
    Handles uploading of predefined voice files (.wav, .mp3).
    Validates files and saves them to the configured predefined voices path.
    """
    logger.info(f"Request to /upload_predefined_voice with {len(files)} file(s).")
    predefined_voices_path = get_predefined_voices_path(ensure_absolute=True)
    uploaded_filenames_successfully: List[str] = []
    upload_errors: List[Dict[str, str]] = []

    for file in files:
        if not file.filename:
            upload_errors.append(
                {"filename": "Unknown", "error": "File received with no filename."}
            )
            logger.warning("Upload attempt for predefined voice with no filename.")
            continue

        safe_filename = utils.sanitize_filename(file.filename)
        destination_path = predefined_voices_path / safe_filename

        try:
            if not (
                safe_filename.lower().endswith(".wav")
                or safe_filename.lower().endswith(".mp3")
            ):
                raise ValueError(
                    "Invalid file type. Only .wav and .mp3 are allowed for predefined voices."
                )

            if destination_path.exists():
                logger.info(
                    f"Predefined voice file '{safe_filename}' already exists. Skipping duplicate upload."
                )
                if safe_filename not in uploaded_filenames_successfully:
                    uploaded_filenames_successfully.append(safe_filename)
                continue

            with open(destination_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(
                f"Successfully saved uploaded predefined voice file to: {destination_path}"
            )
            # Basic validation (can be extended if predefined voices have specific requirements)
            is_valid, validation_msg = utils.validate_reference_audio(
                destination_path, max_duration_sec=None
            )  # No duration limit for predefined
            if not is_valid:
                logger.warning(
                    f"Uploaded predefined voice '{safe_filename}' failed basic validation: {validation_msg}. Deleting."
                )
                destination_path.unlink(missing_ok=True)
                upload_errors.append(
                    {"filename": safe_filename, "error": validation_msg}
                )
            else:
                uploaded_filenames_successfully.append(safe_filename)

        except Exception as e_upload:
            error_msg = f"Error processing predefined voice file '{file.filename}': {str(e_upload)}"
            logger.error(error_msg, exc_info=True)
            upload_errors.append({"filename": file.filename, "error": str(e_upload)})
        finally:
            await file.close()

    all_current_predefined_voices = (
        utils.get_predefined_voices()
    )  # Fetches formatted list
    response_data = {
        "message": f"Processed {len(files)} predefined voice file(s).",
        "uploaded_files": uploaded_filenames_successfully,  # List of raw filenames uploaded
        "all_predefined_voices": all_current_predefined_voices,  # Formatted list for UI
        "errors": upload_errors,
    }
    status_code = (
        200 if not upload_errors or len(uploaded_filenames_successfully) > 0 else 400
    )
    if upload_errors:
        logger.warning(
            f"Upload to /upload_predefined_voice completed with {len(upload_errors)} error(s)."
        )
    return JSONResponse(content=response_data, status_code=status_code)


# --- TTS Generation Endpoint ---


@app.post(
    "/tts",
    tags=["TTS Generation"],
    summary="Generate speech with custom parameters",
    responses={
        200: {
            "content": {"audio/wav": {}, "audio/opus": {}},
            "description": "Successful audio generation.",
        },
        400: {
            "model": ErrorResponse,
            "description": "Invalid request parameters or input.",
        },
        404: {
            "model": ErrorResponse,
            "description": "Required resource not found (e.g., voice file).",
        },
        500: {
            "model": ErrorResponse,
            "description": "Internal server error during generation.",
        },
        503: {
            "model": ErrorResponse,
            "description": "TTS engine not available or model not loaded.",
        },
    },
)
async def custom_tts_endpoint(
    background_tasks: BackgroundTasks,
    text: str = Form(..., description="Text to synthesize"),
    voice_mode: Literal["predefined", "clone"] = Form("predefined"),
    predefined_voice_id: Optional[str] = Form(None),
    reference_audio_file: Optional[UploadFile] = File(None),
    reference_audio_url: Optional[str] = Form(None),
    output_format: str = Form("wav"),
    split_text: bool = Form(True),
    chunk_size: int = Form(320),
    temperature: Optional[float] = Form(None),
    exaggeration: Optional[float] = Form(None),
    cfg_weight: Optional[float] = Form(None),
    seed: Optional[int] = Form(None),
    speed_factor: Optional[float] = Form(None),
    language: Optional[str] = Form(None),
    language_id: Optional[str] = Form(None),
):
    """
    Generates speech audio from text using specified parameters.
    Handles various voice modes (predefined, clone) and audio processing options.
    Returns audio as a stream (WAV or Opus).
    """
    perf_monitor = utils.PerformanceMonitor(
        enabled=config_manager.get_bool("server.enable_performance_monitor", False)
    )
    perf_monitor.record("TTS request received")

    if not engine.MODEL_LOADED:
        logger.error("TTS request failed: Model not loaded.")
        raise HTTPException(
            status_code=503,
            detail="TTS engine model is not currently loaded or available.",
        )

    logger.info(
        f"Received /generate request: mode='{voice_mode}', format='{output_format}'"
    )
    logger.debug(
        f"TTS params: seed={seed}, split={split_text}, chunk_size={chunk_size}"
    )
    logger.debug(f"Input text (first 100 chars): '{text[:100]}...'")

    lang_for_num2word = language_id if language_id is not None else language
    if _should_apply_nepali_num2word(text, lang_for_num2word):
        if _NEPALI_NUM2WORD_FN is None:
            logger.warning("Nepali language detected but nepali-num2word is not installed; skipping number normalization")
        else:
            normalized_text = _convert_nepali_numbers_to_words(text)
            _log_num2word_conversion("/tts", text, normalized_text)
            text = normalized_text

    audio_prompt_path_for_engine: Optional[Path] = None
    if voice_mode == "predefined":
        if not predefined_voice_id:
            raise HTTPException(
                status_code=400,
                detail="Missing 'predefined_voice_id' for 'predefined' voice mode.",
            )
        voices_dir = get_predefined_voices_path(ensure_absolute=True)
        potential_path = voices_dir / predefined_voice_id
        if not potential_path.is_file():
            logger.error(f"Predefined voice file not found: {potential_path}")
            raise HTTPException(
                status_code=404,
                detail=f"Predefined voice file '{predefined_voice_id}' not found.",
            )
        audio_prompt_path_for_engine = potential_path
        logger.info(f"Using predefined voice: {predefined_voice_id}")

    elif voice_mode == "clone":
        if reference_audio_file is not None and reference_audio_file.filename:
            import tempfile
            suffix = Path(reference_audio_file.filename).suffix or ".wav"
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            try:
                content = reference_audio_file.file.read()
                temp_file.write(content)
                temp_file.close()
                audio_prompt_path_for_engine = Path(temp_file.name)
                background_tasks.add_task(lambda p: p.unlink(missing_ok=True), audio_prompt_path_for_engine)
                logger.info("Using uploaded reference audio file for cloning")
            except Exception as e:
                logger.error(f"Error reading uploaded file: {e}")
                raise HTTPException(status_code=400, detail="Error processing uploaded reference file")
        elif reference_audio_url:
            downloaded = utils.download_audio_from_url(reference_audio_url)
            if not downloaded:
                raise HTTPException(status_code=400, detail="Error downloading reference audio URL")
            audio_prompt_path_for_engine = downloaded
            background_tasks.add_task(lambda p: p.unlink(missing_ok=True), audio_prompt_path_for_engine)
            logger.info("Using given S3/HTTP URL for cloning")
        else:
            raise HTTPException(
                status_code=400,
                detail="Missing 'reference_audio_file' or 'reference_audio_url' for 'clone' voice mode.",
            )
            
        max_dur = config_manager.get_int("audio_output.max_reference_duration_sec", 30)
        is_valid, msg = utils.validate_reference_audio(audio_prompt_path_for_engine, max_dur)
        if not is_valid:
            raise HTTPException(
                status_code=400, detail=f"Invalid reference audio: {msg}"
            )

    perf_monitor.record("Parameters and voice path resolved")

    all_audio_segments_np: List[np.ndarray] = []
    final_output_sample_rate = (
        get_audio_sample_rate()
    )  # Target SR for the final output file
    engine_output_sample_rate: Optional[int] = (
        None  # SR from the TTS engine (e.g., 24000 Hz)
    )

    chunk_size_to_use = chunk_size if chunk_size is not None else 320
    should_split = split_text or len(text) > (chunk_size_to_use * 1.5)
    if should_split:
        logger.info(f"Splitting text into chunks of size ~{chunk_size_to_use}.")
        text_chunks = utils.chunk_text_by_sentences(text, chunk_size_to_use)
        text_chunks = _enforce_max_chunk_size(text_chunks, chunk_size_to_use)
        perf_monitor.record(f"Text split into {len(text_chunks)} chunks")
    else:
        text_chunks = [text]
        logger.info(
            "Processing text as a single chunk (splitting not enabled or text too short)."
        )
        text_chunks = _enforce_max_chunk_size(text_chunks, chunk_size_to_use)

    if not text_chunks:
        raise HTTPException(
            status_code=400, detail="Text processing resulted in no usable chunks."
        )

    pending_chunks = list(text_chunks)
    completed_chunks = 0
    cfg_weight_to_use = (
        cfg_weight
        if cfg_weight is not None
        else get_gen_default_cfg_weight()
    )
    speed_factor_to_use = (
        speed_factor
        if speed_factor is not None
        else get_gen_default_speed_factor()
    )

    while pending_chunks:
        chunk = pending_chunks.pop(0)
        completed_chunks += 1
        logger.info(f"Synthesizing chunk {completed_chunks}/{len(text_chunks) + len(pending_chunks)}...")

        chunk_audio_tensor = None
        chunk_sr_from_engine = None
        current_processed_audio_tensor = None

        try:
            chunk_audio_tensor, chunk_sr_from_engine = engine.synthesize(
                text=chunk,
                audio_prompt_path=(
                    str(audio_prompt_path_for_engine)
                    if audio_prompt_path_for_engine
                    else None
                ),
                temperature=(
                    temperature
                    if temperature is not None
                    else get_gen_default_temperature()
                ),
                exaggeration=(
                    exaggeration
                    if exaggeration is not None
                    else get_gen_default_exaggeration()
                ),
                cfg_weight=(
                    cfg_weight_to_use
                ),
                seed=(
                    seed if seed is not None else get_gen_default_seed()
                ),
                language_id=language_id,
            )
            perf_monitor.record(f"Engine synthesized chunk {completed_chunks}")

            if chunk_audio_tensor is None or chunk_sr_from_engine is None:
                sub_chunks = _split_chunk_for_retry(chunk, min_chunk_size=50)
                if sub_chunks:
                    logger.warning(
                        f"Chunk synthesis failed; splitting into {len(sub_chunks)} smaller chunks and retrying."
                    )
                    pending_chunks = sub_chunks + pending_chunks
                    completed_chunks -= 1
                    continue

                error_detail = f"TTS engine failed to synthesize audio for chunk {completed_chunks}."
                logger.error(error_detail)
                raise HTTPException(status_code=500, detail=error_detail)

            if engine_output_sample_rate is None:
                engine_output_sample_rate = chunk_sr_from_engine
            elif engine_output_sample_rate != chunk_sr_from_engine:
                logger.warning(
                    f"Inconsistent sample rate from engine: chunk {completed_chunks} ({chunk_sr_from_engine}Hz) "
                    f"differs from previous ({engine_output_sample_rate}Hz). Using first chunk's SR."
                )

            current_processed_audio_tensor = chunk_audio_tensor

            if speed_factor_to_use != 1.0:
                current_processed_audio_tensor, _ = utils.apply_speed_factor(
                    current_processed_audio_tensor,
                    chunk_sr_from_engine,
                    speed_factor_to_use,
                )
                perf_monitor.record(f"Speed factor applied to chunk {completed_chunks}")

            processed_audio_np = current_processed_audio_tensor.cpu().numpy().squeeze()
            all_audio_segments_np.append(processed_audio_np)

        except HTTPException as http_exc:
            raise http_exc
        except Exception as e_chunk:
            error_detail = f"Error processing audio chunk {completed_chunks}: {str(e_chunk)}"
            logger.error(error_detail, exc_info=True)

            sub_chunks = _split_chunk_for_retry(chunk, min_chunk_size=50)
            if sub_chunks:
                logger.warning(
                    f"Chunk raised exception; splitting into {len(sub_chunks)} smaller chunks for retry."
                )
                pending_chunks = sub_chunks + pending_chunks
                completed_chunks -= 1
                continue

            raise HTTPException(status_code=500, detail=error_detail)
        finally:
            if chunk_audio_tensor is not None:
                del chunk_audio_tensor
            if current_processed_audio_tensor is not None:
                del current_processed_audio_tensor
            _clear_cuda_memory_after_generate()

    if not all_audio_segments_np:
        logger.error("No audio segments were successfully generated.")
        raise HTTPException(
            status_code=500, detail="Audio generation resulted in no output."
        )

    if engine_output_sample_rate is None:
        logger.error("Engine output sample rate could not be determined.")
        raise HTTPException(
            status_code=500, detail="Failed to determine engine sample rate."
        )

    try:
        # ### MODIFICATION START ###
        # Concatenate all raw chunks with crossfade to avoid boundary artifacts.
        if len(all_audio_segments_np) > 1:
            crossfade_ms = 50  # 50ms crossfade between chunks
            crossfade_samples = int(crossfade_ms / 1000.0 * engine_output_sample_rate) if engine_output_sample_rate else 0
            final_audio_np = all_audio_segments_np[0]
            for seg in all_audio_segments_np[1:]:
                overlap = min(crossfade_samples, len(final_audio_np), len(seg))
                if overlap > 0:
                    fade_out = np.linspace(1.0, 0.0, overlap, dtype=final_audio_np.dtype)
                    fade_in = np.linspace(0.0, 1.0, overlap, dtype=seg.dtype)
                    # Blend the overlapping region
                    blended = final_audio_np[-overlap:] * fade_out + seg[:overlap] * fade_in
                    final_audio_np = np.concatenate([final_audio_np[:-overlap], blended, seg[overlap:]])
                else:
                    final_audio_np = np.concatenate([final_audio_np, seg])
        else:
            final_audio_np = all_audio_segments_np[0]
        perf_monitor.record("All audio chunks processed and concatenated (crossfaded)")

        # Now, apply all audio processing to the COMPLETE audio clip.
        if config_manager.get_bool("audio_processing.enable_silence_trimming", False):
            final_audio_np = utils.trim_lead_trail_silence(
                final_audio_np, engine_output_sample_rate
            )
            perf_monitor.record(f"Global silence trim applied")

        if config_manager.get_bool(
            "audio_processing.enable_internal_silence_fix", False
        ):
            final_audio_np = utils.fix_internal_silence(
                final_audio_np, engine_output_sample_rate
            )
            perf_monitor.record(f"Global internal silence fix applied")

        if (
            config_manager.get_bool("audio_processing.enable_unvoiced_removal", False)
            and utils.PARSELMOUTH_AVAILABLE
        ):
            final_audio_np = utils.remove_long_unvoiced_segments(
                final_audio_np, engine_output_sample_rate
            )
            perf_monitor.record(f"Global unvoiced removal applied")
        # ### MODIFICATION END ###

    except ValueError as e_concat:
        logger.error(f"Audio concatenation failed: {e_concat}", exc_info=True)
        for idx, seg in enumerate(all_audio_segments_np):
            logger.error(f"Segment {idx} shape: {seg.shape}, dtype: {seg.dtype}")
        raise HTTPException(
            status_code=500, detail=f"Audio concatenation error: {e_concat}"
        )

    output_format_str = (
        output_format if output_format else get_audio_output_format()
    )

    encoded_audio_bytes = utils.encode_audio(
        audio_array=final_audio_np,
        sample_rate=engine_output_sample_rate,
        output_format=output_format_str,
        target_sample_rate=final_output_sample_rate,
    )
    perf_monitor.record(
        f"Final audio encoded to {output_format_str} (target SR: {final_output_sample_rate}Hz from engine SR: {engine_output_sample_rate}Hz)"
    )

    if encoded_audio_bytes is None or len(encoded_audio_bytes) < 100:
        logger.error(
            f"Failed to encode final audio to format: {output_format_str} or output is too small ({len(encoded_audio_bytes or b'')} bytes)."
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to encode audio to {output_format_str} or generated invalid audio.",
        )

    media_type = f"audio/{output_format_str}"
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    suggested_filename_base = f"tts_output_{timestamp_str}"
    download_filename = utils.sanitize_filename(
        f"{suggested_filename_base}.{output_format_str}"
    )
    headers = {"Content-Disposition": f'attachment; filename="{download_filename}"'}

    logger.info(
        f"Successfully generated audio: {download_filename}, {len(encoded_audio_bytes)} bytes, type {media_type}."
    )
    logger.debug(perf_monitor.report())

    return StreamingResponse(
        io.BytesIO(encoded_audio_bytes), media_type=media_type, headers=headers
    )


@app.post(
    "/tts/warmup",
    tags=["TTS Generation"],
    summary="Warm up the TTS model (voice conditioning + dummy generation)",
)
async def tts_warmup_endpoint(voice_filename: Optional[str] = Form(None)):
    """
    Triggers engine.warmup_model() to pre-cache voice conditioning and run
    a short dummy generation to reduce first-request latency.

    Optionally accepts a `voice_filename` (predefined voice filename) to
    use for conditioning. If omitted, the engine will pick the default voice.
    Returns the warmup timing/status dictionary from `engine.warmup_model()`.
    """
    logger.info("Request received for /tts/warmup")
    voice_path = None
    try:
        if voice_filename:
            voices_dir = get_predefined_voices_path(ensure_absolute=True)
            potential = voices_dir / voice_filename
            if not potential.is_file():
                raise HTTPException(
                    status_code=404,
                    detail=f"Predefined voice file '{voice_filename}' not found.",
                )
            voice_path = str(potential)

        result = engine.warmup_model(voice_path)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during warmup endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# --- STT Generation Endpoint ---

@app.post(
    "/stt",
    response_model=STTResponse,
    tags=["STT Generation"],
    summary="Transcribe speech to text",
    responses={
        200: {
            "model": STTResponse,
            "description": "Successful speech-to-text transcription.",
        },
        400: {
            "model": ErrorResponse,
            "description": "Invalid request parameters or audio file.",
        },
        503: {
            "model": ErrorResponse,
            "description": "STT engine not available or model not loaded.",
        },
    },
)
async def speech_to_text_endpoint(
    audio_file: UploadFile = File(..., description="Audio file to transcribe (.wav, .mp3, .m4a, .flac)"),
    language: Optional[str] = Form(None, description="Language code or None for auto-detection"),
    lang: Optional[str] = Form(None, description="Alias for language code"),
    stt_engine: STTEngine = Depends(get_stt_engine)
):
    """
    Transcribes speech from an uploaded audio file to text.
    Supports common audio formats and multiple languages.
    """
    if not stt_engine.model_loaded:
        logger.error("STT request failed: Model not loaded.")
        raise HTTPException(
            status_code=503,
            detail="STT engine model is not currently loaded or available.",
        )
    
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No audio file provided.")
    
    # Validate file extension
    allowed_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    file_ext = Path(audio_file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {file_ext}. Supported: {', '.join(allowed_extensions)}"
        )
    
    requested_language = language if language not in (None, "") else lang
    normalized_language = None
    if requested_language is not None:
        candidate = str(requested_language).strip()
        normalized_language = candidate if candidate else None

    logger.info(
        "Received STT request for file: %s | language=%s",
        audio_file.filename,
        normalized_language if normalized_language else "auto",
    )
    
    # Save uploaded file temporarily
    temp_audio_path = get_output_path() / f"temp_stt_{uuid.uuid4().hex[:8]}{file_ext}"
    
    try:
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        # Transcribe the audio using effective language from payload/config.
        transcription = stt_engine.transcribe_file_with_metadata(
            str(temp_audio_path),
            normalized_language,
        )

        if transcription is None:
            raise HTTPException(
                status_code=500,
                detail="Transcription failed. Please check audio file format and content."
            )

        transcribed_text = str(transcription.get("text", "")).strip()
        resolved_language = transcription.get("language") or normalized_language
        
        logger.info(f"STT transcription successful. Text length: {len(transcribed_text)} characters")
        
        return STTResponse(
            text=transcribed_text,
            language=resolved_language,
            duration=None  # Could add duration calculation if needed
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during STT processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"STT processing error: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_audio_path.exists():
            temp_audio_path.unlink(missing_ok=True)
        await audio_file.close()


@app.post(
    "/generate",
    tags=["TTS Generation"],
    summary="Generate speech with reference audio and advanced parameters",
    responses={
        200: {
            "content": {"audio/wav": {}},
            "description": "Successful audio generation.",
        },
        400: {
            "model": ErrorResponse,
            "description": "Invalid request parameters or input.",
        },
        503: {
            "model": ErrorResponse,
            "description": "TTS engine not available or model not loaded.",
        },
    },
)
async def generate_speech_endpoint(
    background_tasks: BackgroundTasks,
    text: str = Form(..., description="Text to convert to speech"),
    reference_audio: Optional[UploadFile] = File(None, description="Optional reference audio file for voice cloning"),
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
    chunk_size: int = Form(320, description="Target chunk size for text splitting (50-500)", ge=50, le=500),
    language_id: Optional[str] = Form('ne', description="Language code for multilingual model (e.g., 'en', 'fr', 'es', 'zh')"),
):
    """
    Generates speech audio from text with advanced parameters and optional reference audio for voice cloning.
    Returns audio as WAV format.
    """
    if not engine.MODEL_LOADED:
        logger.error("Generate request failed: Model not loaded.")
        raise HTTPException(
            status_code=503,
            detail="TTS engine model is not currently loaded or available.",
        )
    if engine.has_cuda_device_assert():
        raise HTTPException(status_code=503, detail=_cuda_assert_http_detail())
    
    logger.info(f"Received /generate request with text: '{text[:50]}...', reference_audio: '{reference_audio.filename if reference_audio else 'None'}', language_id: {language_id}, speed factor: {speed_factor}, cfg_weight: {cfg_weight}")

    if _should_apply_nepali_num2word(text, language_id):
        if _NEPALI_NUM2WORD_FN is None:
            logger.warning("Nepali language detected but nepali-num2word is not installed; skipping number normalization")
        else:
            normalized_text = _convert_nepali_numbers_to_words(text)
            _log_num2word_conversion("/generate", text, normalized_text)
            text = normalized_text
    
    # Handle reference audio if provided
    audio_prompt_path: Optional[Path] = None
    temp_ref_audio_path: Optional[Path] = None
    
    if reference_audio and reference_audio.filename:
        # Validate file extension
        allowed_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
        file_ext = Path(reference_audio.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported reference audio format: {file_ext}. Supported: {', '.join(allowed_extensions)}"
            )
        
        # Save reference audio temporarily
        temp_ref_audio_path = get_output_path() / f"temp_ref_{uuid.uuid4().hex[:8]}{file_ext}"

        try:
            with open(temp_ref_audio_path, "wb") as buffer:
                shutil.copyfileobj(reference_audio.file, buffer)

            # Validate reference audio
            max_dur = config_manager.get_int("audio_output.max_reference_duration_sec", 30)

            audio_prompt_path = temp_ref_audio_path
            background_tasks.add_task(lambda p: p.unlink(missing_ok=True), temp_ref_audio_path)
            logger.info(f"Using uploaded reference audio: {reference_audio.filename}")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing reference audio: {e}", exc_info=True)
            if temp_ref_audio_path and temp_ref_audio_path.exists():
                temp_ref_audio_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Failed to process reference audio: {str(e)}")
        finally:
            await reference_audio.close()
    
    # Initialize audio collection and sample rate tracking
    all_audio_segments_np: List[np.ndarray] = []
    engine_output_sample_rate: Optional[int] = None
    
    # Handle text chunking
    should_split = split_text or len(text) > (chunk_size * 1.5)
    if should_split:
        logger.info(f"Splitting text into chunks of size ~{chunk_size}.")
        text_chunks = utils.chunk_text_by_sentences(text, chunk_size)
        text_chunks = _enforce_max_chunk_size(text_chunks, chunk_size)
    else:
        text_chunks = [text]
        logger.info("Processing text as a single chunk (splitting not enabled and text short).")
        text_chunks = _enforce_max_chunk_size(text_chunks, chunk_size)
    
    if not text_chunks:
        raise HTTPException(status_code=400, detail="Text processing resulted in no usable chunks.")
    
    # Process each chunk with adaptive split-and-retry for OOM resilience
    pending_chunks = list(text_chunks)
    completed_chunks = 0
    max_adaptive_splits = 4
    cfg_weight_to_use = (
        cfg_weight
        if cfg_weight is not None
        else get_gen_default_cfg_weight()
    )
    speed_factor_to_use = (
        speed_factor
        if speed_factor is not None
        else get_gen_default_speed_factor()
    )

    while pending_chunks:
        chunk = pending_chunks.pop(0)
        completed_chunks += 1
        logger.info(f"Synthesizing chunk {completed_chunks}/{len(text_chunks) + len(pending_chunks)}...")

        # Track adaptive splitting depth to avoid infinite retries on pathological input
        split_attempts = 0

        while True:
            chunk_audio_tensor = None
            chunk_sr_from_engine = None
            current_processed_audio_tensor = None

            if split_attempts > max_adaptive_splits:
                raise HTTPException(
                    status_code=500,
                    detail="Text chunk is too complex/long to synthesize without OOM. Please reduce chunk_size or split text manually."
                )

            if split_attempts > 0:
                logger.warning(
                    f"Retrying chunk with smaller split (attempt {split_attempts}/{max_adaptive_splits})."
                )

            logger.debug(f"Chunk length: {len(chunk)} chars")

            try:
                chunk_audio_tensor, chunk_sr_from_engine = engine.synthesize(
                    text=chunk,
                    audio_prompt_path=str(audio_prompt_path) if audio_prompt_path else None,
                    temperature=temperature,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight_to_use,
                    seed=seed,
                    language_id=language_id,
                )

                if chunk_audio_tensor is None or chunk_sr_from_engine is None:
                    # Likely synthesis failure (including possible CUDA OOM in engine layer).
                    if engine.has_cuda_device_assert():
                        raise HTTPException(status_code=503, detail=_cuda_assert_http_detail())

                    sub_chunks = _split_chunk_for_retry(chunk, min_chunk_size=50)
                    if sub_chunks:
                        logger.warning(
                            f"Chunk synthesis failed; splitting into {len(sub_chunks)} smaller chunks and retrying."
                        )
                        pending_chunks = sub_chunks + pending_chunks
                        completed_chunks -= 1
                        break

                    error_detail = f"TTS engine failed to synthesize audio for chunk {completed_chunks}."
                    logger.error(error_detail)
                    raise HTTPException(status_code=500, detail=error_detail)

                if engine_output_sample_rate is None:
                    engine_output_sample_rate = chunk_sr_from_engine
                elif engine_output_sample_rate != chunk_sr_from_engine:
                    logger.warning(
                        f"Inconsistent sample rate from engine: chunk {completed_chunks} ({chunk_sr_from_engine}Hz) "
                        f"differs from previous ({engine_output_sample_rate}Hz). Using first chunk's SR."
                    )

                current_processed_audio_tensor = chunk_audio_tensor
                if speed_factor_to_use != 1.0:
                    current_processed_audio_tensor, _ = utils.apply_speed_factor(
                        current_processed_audio_tensor,
                        chunk_sr_from_engine,
                        speed_factor_to_use,
                    )

                # Convert to numpy and collect
                processed_audio_np = current_processed_audio_tensor.cpu().numpy().squeeze()
                all_audio_segments_np.append(processed_audio_np)
                break

            except HTTPException as http_exc:
                raise http_exc
            except Exception as e_chunk:
                error_detail = f"Error processing audio chunk {completed_chunks}: {str(e_chunk)}"
                logger.error(error_detail, exc_info=True)

                # Attempt adaptive split retry before hard-failing.
                sub_chunks = _split_chunk_for_retry(chunk, min_chunk_size=50)
                if sub_chunks:
                    split_attempts += 1
                    logger.warning(
                        f"Chunk raised exception; splitting into {len(sub_chunks)} smaller chunks for retry."
                    )
                    pending_chunks = sub_chunks + pending_chunks
                    completed_chunks -= 1
                    break

                raise HTTPException(status_code=500, detail=error_detail)
            finally:
                # Free per-chunk intermediates aggressively to reduce VRAM growth on long texts.
                if chunk_audio_tensor is not None:
                    del chunk_audio_tensor
                if current_processed_audio_tensor is not None:
                    del current_processed_audio_tensor
                _clear_cuda_memory_after_generate()
    
    if not all_audio_segments_np:
        logger.error("No audio segments were successfully generated.")
        raise HTTPException(status_code=500, detail="Audio generation resulted in no output.")
    
    if engine_output_sample_rate is None:
        logger.error("Engine output sample rate could not be determined.")
        raise HTTPException(status_code=500, detail="Failed to determine engine sample rate.")
    
    try:
        # Concatenate buffered chunk audio into one final clip before sending.
        if len(all_audio_segments_np) > 1:
            crossfade_ms = 35
            crossfade_samples = int(crossfade_ms / 1000.0 * engine_output_sample_rate) if engine_output_sample_rate else 0
            final_audio_np = all_audio_segments_np[0]
            for seg in all_audio_segments_np[1:]:
                overlap = min(crossfade_samples, len(final_audio_np), len(seg))
                if overlap > 0:
                    fade_out = np.linspace(1.0, 0.0, overlap, dtype=final_audio_np.dtype)
                    fade_in = np.linspace(0.0, 1.0, overlap, dtype=seg.dtype)
                    blended = final_audio_np[-overlap:] * fade_out + seg[:overlap] * fade_in
                    final_audio_np = np.concatenate([final_audio_np[:-overlap], blended, seg[overlap:]])
                else:
                    final_audio_np = np.concatenate([final_audio_np, seg])
        else:
            final_audio_np = all_audio_segments_np[0]
        logger.info(f"All {len(all_audio_segments_np)} chunk audios combined into final output")
        
        # Apply global audio processing
        if config_manager.get_bool("audio_processing.enable_silence_trimming", False):
            final_audio_np = utils.trim_lead_trail_silence(final_audio_np, engine_output_sample_rate)
            logger.debug("Global silence trim applied")
        
        if config_manager.get_bool("audio_processing.enable_internal_silence_fix", False):
            final_audio_np = utils.fix_internal_silence(final_audio_np, engine_output_sample_rate)
            logger.debug("Global internal silence fix applied")
        
        if (
            config_manager.get_bool("audio_processing.enable_unvoiced_removal", False)
            and utils.PARSELMOUTH_AVAILABLE
        ):
            final_audio_np = utils.remove_long_unvoiced_segments(final_audio_np, engine_output_sample_rate)
            logger.debug("Global unvoiced removal applied")
        
        # Encode the audio to WAV format
        encoded_audio = utils.encode_audio(
            audio_array=final_audio_np,
            sample_rate=engine_output_sample_rate,
            output_format="wav",
            target_sample_rate=get_audio_sample_rate(),
        )
        
    except ValueError as e_concat:
        logger.error(f"Audio concatenation failed: {e_concat}", exc_info=True)
        for idx, seg in enumerate(all_audio_segments_np):
            logger.error(f"Segment {idx} shape: {seg.shape}, dtype: {seg.dtype}")
        raise HTTPException(status_code=500, detail=f"Audio concatenation error: {e_concat}")
    except Exception as e:
        logger.error(f"Error during audio processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Audio processing error: {str(e)}")
    
    if encoded_audio is None or len(encoded_audio) < 100:
        raise HTTPException(
            status_code=500, detail="Failed to encode audio or generated invalid audio."
        )
    
    # Prepare response
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    download_filename = utils.sanitize_filename(f"generated_{timestamp_str}.wav")
    headers = {"Content-Disposition": f'attachment; filename="{download_filename}"'}
    
    logger.info(f"Successfully generated audio: {download_filename}, {len(encoded_audio)} bytes")
    
    return StreamingResponse(
        io.BytesIO(encoded_audio), 
        media_type="audio/wav", 
        headers=headers
    )


@app.post("/v1/audio/speech", tags=["OpenAI Compatible"])
async def openai_speech_endpoint(request: OpenAISpeechRequest):
    # Determine the audio prompt path based on the voice parameter
    predefined_voices_path = get_predefined_voices_path(ensure_absolute=True)
    reference_audio_path = get_reference_audio_path(ensure_absolute=True)
    voice_path_predefined = predefined_voices_path / request.voice
    voice_path_reference = reference_audio_path / request.voice

    if voice_path_predefined.is_file():
        audio_prompt_path = voice_path_predefined
    elif voice_path_reference.is_file():
        audio_prompt_path = voice_path_reference
    else:
        raise HTTPException(
            status_code=404, detail=f"Voice file '{request.voice}' not found."
        )

    # Check if the TTS model is loaded
    if not engine.MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="TTS engine model is not currently loaded or available.",
        )
    if engine.has_cuda_device_assert():
        raise HTTPException(status_code=503, detail=_cuda_assert_http_detail())

    try:
        # Use the provided seed or the default
        seed_to_use = (
            request.seed if request.seed is not None else get_gen_default_seed()
        )

        # Synthesize the audio
        audio_tensor, sr = engine.synthesize(
            text=request.input_,
            audio_prompt_path=str(audio_prompt_path),
            temperature=get_gen_default_temperature(),
            exaggeration=get_gen_default_exaggeration(),
            cfg_weight=get_gen_default_cfg_weight(),
            seed=seed_to_use,
            language_id=request.language_id,
        )

        if audio_tensor is None or sr is None:
            raise HTTPException(
                status_code=500, detail="TTS engine failed to synthesize audio."
            )

        # Apply speed factor if not 1.0
        if request.speed != 1.0:
            audio_tensor, _ = utils.apply_speed_factor(audio_tensor, sr, request.speed)

        # Convert tensor to numpy array
        audio_np = audio_tensor.cpu().numpy()

        # Ensure it's 1D
        if audio_np.ndim == 2:
            audio_np = audio_np.squeeze()

        # Encode the audio to the requested format
        encoded_audio = utils.encode_audio(
            audio_array=audio_np,
            sample_rate=sr,
            output_format=request.response_format,
            target_sample_rate=get_audio_sample_rate(),
        )

        if encoded_audio is None:
            raise HTTPException(status_code=500, detail="Failed to encode audio.")

        # Determine the media type
        media_type = f"audio/{request.response_format}"

        # Return the streaming response
        return StreamingResponse(io.BytesIO(encoded_audio), media_type=media_type)

    except Exception as e:
        logger.error(f"Error in openai_speech_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# --- Main Execution ---
if __name__ == "__main__":
    server_host = get_host()
    server_port = get_port()

    logger.info(f"Starting TTS Server directly on http://{server_host}:{server_port}")
    logger.info(
        f"API documentation will be available at http://{server_host}:{server_port}/docs"
    )
    logger.info(f"Web UI will be available at http://{server_host}:{server_port}/")

    import uvicorn

    uvicorn.run(
        "server:app",
        host=server_host,
        port=server_port,
        log_level="info",
        workers=1,
        reload=False,
    )
