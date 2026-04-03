# File: engine.py
# Core TTS model loading and speech generation logic.

import logging
import random
import time
import numpy as np
import torch
from typing import Any, Generator, Optional, Tuple
from pathlib import Path

from chatterbox.tts import ChatterboxTTS  # Main TTS engine class
from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # Multilingual TTS engine class
from chatterbox.models.s3gen.const import (
    S3GEN_SR,
)  # Default sample rate from the engine

# Import the singleton config_manager
from config import config_manager

logger = logging.getLogger(__name__)

# --- Global Module Variables ---
chatterbox_model: Optional[ChatterboxTTS] = None
MODEL_LOADED: bool = False
model_device: Optional[str] = (
    None  # Stores the resolved device string ('cuda' or 'cpu')
)

# Voice conditioning cache: avoids re-running prepare_conditionals for the same voice
# Key: audio_prompt_path (str), Value: True (conds are on the model instance)
_last_prepared_voice: Optional[str] = None
_warmup_done: bool = False


def set_seed(seed_value: int):
    """
    Sets the seed for torch, random, and numpy for reproducibility.
    This is called if a non-zero seed is provided for generation.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    logger.info(f"Global seed set to: {seed_value}")


def _test_cuda_functionality() -> bool:
    """
    Tests if CUDA is actually functional, not just available.

    Returns:
        bool: True if CUDA works, False otherwise.
    """
    if not torch.cuda.is_available():
        return False

    try:
        test_tensor = torch.tensor([1.0])
        test_tensor = test_tensor.cuda()
        test_tensor = test_tensor.cpu()
        return True
    except Exception as e:
        logger.warning(f"CUDA functionality test failed: {e}")
        return False


def _test_mps_functionality() -> bool:
    """
    Tests if MPS is actually functional, not just available.

    Returns:
        bool: True if MPS works, False otherwise.
    """
    if not torch.backends.mps.is_available():
        return False

    try:
        test_tensor = torch.tensor([1.0])
        test_tensor = test_tensor.to("mps")
        test_tensor = test_tensor.cpu()
        return True
    except Exception as e:
        logger.warning(f"MPS functionality test failed: {e}")
        return False


def warmup_model(voice_path: Optional[str] = None) -> dict:
    """
    Warm up the TTS model to minimize first-request latency.
    
    Runs:
    1. prepare_conditionals with the default/specified voice (caches voice embeddings)
    2. A short dummy generation to trigger T3 model compilation + CUDA kernel caching
    
    Args:
        voice_path: Path to voice file to pre-cache. If None, uses default voice.
    
    Returns:
        dict with warmup timing info
    """
    global chatterbox_model, _last_prepared_voice, _warmup_done
    
    if not MODEL_LOADED or chatterbox_model is None:
        logger.warning("Cannot warmup: model not loaded")
        return {"status": "error", "message": "Model not loaded"}
    
    timings = {}
    
    # 1. Pre-cache voice conditioning
    if voice_path is None:
        # Find default voice
        from config import get_predefined_voices_path
        voices_dir = get_predefined_voices_path(ensure_absolute=True)
        default_voices = list(voices_dir.glob("*.wav"))
        if default_voices:
            voice_path = str(default_voices[0])
            logger.info(f"Warmup: Using default voice: {voice_path}")
        else:
            logger.warning("Warmup: No default voice files found, skipping conditioning warmup")
    
    if voice_path and hasattr(chatterbox_model, 'prepare_conditionals'):
        t0 = time.time()
        try:
            chatterbox_model.prepare_conditionals(voice_path, exaggeration=0.5)
            _last_prepared_voice = voice_path
            timings["prepare_conditionals_ms"] = (time.time() - t0) * 1000
            logger.info(f"Warmup: Voice conditioning cached in {timings['prepare_conditionals_ms']:.0f}ms")
        except Exception as e:
            logger.error(f"Warmup: prepare_conditionals failed: {e}")
            timings["prepare_conditionals_error"] = str(e)
    
    # 2. Dummy generation to compile T3 model + warm CUDA kernels
    t1 = time.time()
    try:
        # Detect if multilingual model (requires language_id)
        import inspect
        gen_kwargs = {"text": "Hello.", "temperature": 0.8}
        if hasattr(chatterbox_model, 'generate_stream'):
            sig = inspect.signature(chatterbox_model.generate_stream)
            if 'language_id' in sig.parameters:
                gen_kwargs["language_id"] = "en"
            gen_kwargs["chunk_size"] = 25
            # Use streaming to also warm the streaming path
            for audio_chunk, metrics in chatterbox_model.generate_stream(**gen_kwargs):
                pass  # Just run through to trigger compilation
        else:
            sig = inspect.signature(chatterbox_model.generate)
            if 'language_id' in sig.parameters:
                gen_kwargs["language_id"] = "en"
            chatterbox_model.generate(**gen_kwargs)
        timings["dummy_generation_ms"] = (time.time() - t1) * 1000
        logger.info(f"Warmup: Dummy generation completed in {timings['dummy_generation_ms']:.0f}ms")
    except Exception as e:
        logger.error(f"Warmup: Dummy generation failed: {e}")
        timings["dummy_generation_error"] = str(e)
    
    _warmup_done = True
    timings["status"] = "ok"
    total_ms = sum(v for k, v in timings.items() if k.endswith("_ms"))
    timings["total_ms"] = total_ms
    logger.info(f"Warmup: Complete in {total_ms:.0f}ms")
    
    return timings


def ensure_voice_prepared(audio_prompt_path: Optional[str]) -> Optional[str]:
    """
    Ensure voice conditioning is prepared, using cache when possible.
    
    If the same voice was already prepared (cached on the model instance),
    returns None to signal generate_stream should NOT pass audio_prompt_path
    (so it reuses self.conds). Otherwise returns the path unchanged.
    
    Args:
        audio_prompt_path: Voice file path to prepare
        
    Returns:
        None if voice is already cached (generate_stream will reuse self.conds),
        or the original path if new conditioning is needed.
    """
    global chatterbox_model, _last_prepared_voice
    
    if audio_prompt_path is None:
        return None
    
    # Check if this voice is already the active conditioning
    if (_last_prepared_voice is not None 
        and _last_prepared_voice == audio_prompt_path 
        and chatterbox_model is not None 
        and chatterbox_model.conds is not None):
        logger.info(f"Voice conditioning cache HIT: {Path(audio_prompt_path).name}")
        return None  # Already prepared — tell caller to skip audio_prompt_path
    
    # New voice — will be prepared by generate_stream, update cache tracker
    logger.info(f"Voice conditioning cache MISS: {Path(audio_prompt_path).name} (previous: {Path(_last_prepared_voice).name if _last_prepared_voice else 'None'})")
    _last_prepared_voice = audio_prompt_path
    return audio_prompt_path


def is_warmed_up() -> bool:
    """Check if warmup has been performed."""
    return _warmup_done


def load_model() -> bool:
    """
    Loads the TTS model.
    This version directly attempts to load from the Hugging Face repository (or its cache)
    using `from_pretrained`, bypassing the local `paths.model_cache` directory.
    Updates global variables `chatterbox_model`, `MODEL_LOADED`, and `model_device`.

    Returns:
        bool: True if the model was loaded successfully, False otherwise.
    """
    global chatterbox_model, MODEL_LOADED, model_device

    if MODEL_LOADED:
        logger.info("TTS model is already loaded.")
        return True
    try:
        # Determine processing device with robust CUDA detection and intelligent fallback
        device_setting = config_manager.get_string("tts_engine.device", "auto")

        if device_setting == "auto":
            if _test_cuda_functionality():
                resolved_device_str = "cuda"
                logger.info("CUDA functionality test passed. Using CUDA.")
            elif _test_mps_functionality():
                resolved_device_str = "mps"
                logger.info("MPS functionality test passed. Using MPS.")
            else:
                resolved_device_str = "cpu"
                logger.info("CUDA and MPS not functional or not available. Using CPU.")

        elif device_setting == "cuda":
            if _test_cuda_functionality():
                resolved_device_str = "cuda"
                logger.info("CUDA requested and functional. Using CUDA.")
            else:
                resolved_device_str = "cpu"
                logger.warning(
                    "CUDA was requested in config but functionality test failed. "
                    "PyTorch may not be compiled with CUDA support. "
                    "Automatically falling back to CPU."
                )

        elif device_setting == "mps":
            if _test_mps_functionality():
                resolved_device_str = "mps"
                logger.info("MPS requested and functional. Using MPS.")
            else:
                resolved_device_str = "cpu"
                logger.warning(
                    "MPS was requested in config but functionality test failed. "
                    "PyTorch may not be compiled with MPS support. "
                    "Automatically falling back to CPU."
                )

        elif device_setting == "cpu":
            resolved_device_str = "cpu"
            logger.info("CPU device explicitly requested in config. Using CPU.")

        else:
            logger.warning(
                f"Invalid device setting '{device_setting}' in config. "
                f"Defaulting to auto-detection."
            )
            if _test_cuda_functionality():
                resolved_device_str = "cuda"
            elif _test_mps_functionality():
                resolved_device_str = "mps"
            else:
                resolved_device_str = "cpu"
            logger.info(f"Auto-detection resolved to: {resolved_device_str}")

        model_device = resolved_device_str
        logger.info(f"Final device selection: {model_device}")

        # Get configured model_repo_id for logging and context,
        # though from_pretrained might use its own internal default if not overridden.
        model_repo_id_config = config_manager.get_string(
            "model.repo_id", "ResembleAI/chatterbox"
        )
        local_model_path_cfg = config_manager.get_string("model.local_path", "").strip()
        use_local_model = local_model_path_cfg.lower() not in {"", "none", "null"}

        # Check if multilingual model should be used
        use_multilingual = config_manager.get_bool("model.use_multilingual", False)
        
        # Check if multilingual support is available (only in chatterbox-vllm package)

        if use_multilingual:
            logger.info("Attempting to load MULTILINGUAL model.")
        else:
            logger.info(
                f"Attempting to load STANDARD model directly using from_pretrained (expected from Hugging Face repository: {model_repo_id_config} or library default)."
            )

        local_model_dir: Optional[Path] = None
        if use_local_model:
            configured_path = Path(local_model_path_cfg).expanduser()
            local_model_dir = configured_path if configured_path.is_absolute() else (Path.cwd() / configured_path)
            local_model_dir = local_model_dir.resolve()

            if not local_model_dir.exists() or not local_model_dir.is_dir():
                logger.error(
                    f"Configured model.local_path does not exist or is not a directory: {local_model_dir}"
                )
                chatterbox_model = None
                MODEL_LOADED = False
                return False

            logger.info(f"Using local finetuned model path: {local_model_dir}")
        
        logger.info(f"Loading TTS model... on device {model_device} and multilingual={use_multilingual}")

        try:
            # Load either multilingual or standard model based on configuration
            if use_multilingual:
                if local_model_dir is not None:
                    chatterbox_model = ChatterboxMultilingualTTS.from_local(
                        ckpt_dir=local_model_dir,
                        device=model_device,
                    )
                    logger.info(
                        f"Successfully loaded local MULTILINGUAL TTS model from '{local_model_dir}' on {model_device}."
                    )
                else:
                    chatterbox_model = ChatterboxMultilingualTTS.from_pretrained(device=model_device)
                    logger.info(
                        f"Successfully loaded MULTILINGUAL TTS model on {model_device} (supports 23 languages)."
                    )
            else:
                if local_model_dir is not None:
                    chatterbox_model = ChatterboxTTS.from_local(
                        ckpt_dir=local_model_dir,
                        device=model_device,
                    )
                    logger.info(
                        f"Successfully loaded local STANDARD TTS model from '{local_model_dir}' on {model_device}."
                    )
                else:
                    # Directly use from_pretrained. This will utilize the standard Hugging Face cache.
                    # The ChatterboxTTS.from_pretrained method handles downloading if the model is not in the cache.
                    chatterbox_model = ChatterboxTTS.from_pretrained(device=model_device)
                    # The actual repo ID used by from_pretrained is often internal to the library,
                    # but logging the configured one provides user context.
                    logger.info(
                        f"Successfully loaded STANDARD TTS model using from_pretrained on {model_device} (expected from '{model_repo_id_config}' or library default)."
                    )
        except Exception as e_hf:
            logger.error(
                f"Failed to load model using from_pretrained (expected from '{model_repo_id_config}' or library default): {e_hf}",
                exc_info=True,
            )
            chatterbox_model = None
            MODEL_LOADED = False
            return False

        MODEL_LOADED = True
        if chatterbox_model:
            logger.info(
                f"TTS Model loaded successfully on {model_device}. Engine sample rate: {chatterbox_model.sr} Hz."
            )
        else:
            logger.error(
                "Model loading sequence completed, but chatterbox_model is None. This indicates an unexpected issue."
            )
            MODEL_LOADED = False
            return False

        return True

    except Exception as e:
        logger.error(
            f"An unexpected error occurred during model loading: {e}", exc_info=True
        )
        chatterbox_model = None
        MODEL_LOADED = False
        return False


def synthesize(
    text: str,
    audio_prompt_path: Optional[str] = None,
    temperature: float = 0.8,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    seed: int = 0,
    language_id: Optional[str] = None,
) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    """
    Synthesizes audio from text using the loaded TTS model.

    Args:
        text: The text to synthesize.
        audio_prompt_path: Path to an audio file for voice cloning or predefined voice.
        temperature: Controls randomness in generation.
        exaggeration: Controls expressiveness.
        cfg_weight: Classifier-Free Guidance weight.
        seed: Random seed for generation. If 0, default randomness is used.
              If non-zero, a global seed is set for reproducibility.
        language_id: Language code for multilingual model (e.g., 'en', 'fr', 'es', 'zh').
                     Only used if multilingual model is loaded. If None, defaults to config.

    Returns:
        A tuple containing the audio waveform (torch.Tensor) and the sample rate (int),
        or (None, None) if synthesis fails.
    """
    global chatterbox_model

    if not MODEL_LOADED or chatterbox_model is None:
        logger.error("TTS model is not loaded. Cannot synthesize audio.")
        return None, None
    logger.info(f"\nSynthesizing text: {text} language_id: {language_id}")

    try:
        # Set seed globally if a specific seed value is provided and is non-zero.
        if seed != 0:
            logger.info(f"Applying user-provided seed for generation: {seed}")
            set_seed(seed)
        else:
            logger.info(
                "Using default (potentially random) generation behavior as seed is 0."
            )

        # Get language_id from config if not provided
        if language_id is None:
            language_id = config_manager.get_string("generation_defaults.language", "en")
        
        # Use voice conditioning cache to skip redundant prepare_conditionals
        effective_audio_prompt = ensure_voice_prepared(audio_prompt_path)
        
        logger.debug(
            f"Synthesizing with params: audio_prompt='{audio_prompt_path}' (effective='{effective_audio_prompt}'), temp={temperature}, "
            f"exag={exaggeration}, cfg_weight={cfg_weight}, language_id={language_id}, seed_applied_globally_if_nonzero={seed}"
        )

        # Call the core model's generate method
        # Check if the generate method supports language_id parameter
        import inspect
        generate_signature = inspect.signature(chatterbox_model.generate)
        supports_language_id = 'language_id' in generate_signature.parameters
        
        if supports_language_id:
            # Multilingual model - pass language_id
            wav_tensor = chatterbox_model.generate(
                text=text,
                audio_prompt_path=effective_audio_prompt,
                temperature=temperature,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                language_id=language_id,
            )
        else:
            # Standard model - don't pass language_id
            if language_id and language_id != "en":
                logger.warning(
                    f"Language '{language_id}' requested but standard chatterbox-tts only supports English. "
                    "To use multilingual TTS, install chatterbox-vllm package."
                )
            wav_tensor = chatterbox_model.generate(
                text=text,
                audio_prompt_path=effective_audio_prompt,
                temperature=temperature,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )

        # The ChatterboxTTS.generate method already returns a CPU tensor.
        return wav_tensor, chatterbox_model.sr

    except Exception as e:
        logger.error(f"Error during TTS synthesis: {e}", exc_info=True)
        return None, None



def synthesize_stream(
    text: str,
    audio_prompt_path: Optional[str] = None,
    temperature: float = 0.8,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    seed: int = 0,
    chunk_size: int = 25,
    language_id: Optional[str] = None,
) -> Generator[Tuple[Optional[torch.Tensor], Optional[Any]], None, None]:
    """
    Synthesizes audio stream from text using the loaded TTS model.

    Yields:
        Tuple of (audio_chunk, metrics)
    """
    global chatterbox_model

    if not MODEL_LOADED or chatterbox_model is None:
        logger.error("TTS model is not loaded. Cannot synthesize audio.")
        yield None, None
        return

    logger.info(f"\nSynthesizing stream: {text[:50]}... language_id: {language_id}")

    try:
        if seed != 0:
            set_seed(seed)

        if language_id is None:
            language_id = config_manager.get_string("generation_defaults.language", "en")

        # Use voice conditioning cache to skip redundant prepare_conditionals
        effective_audio_prompt = ensure_voice_prepared(audio_prompt_path)

        # Check for language_id support
        import inspect

        # Check if generate_stream exists; if not, fall back to generate()
        use_streaming = hasattr(chatterbox_model, 'generate_stream')

        if use_streaming:
            generate_stream_signature = inspect.signature(chatterbox_model.generate_stream)
            supports_language_id = 'language_id' in generate_stream_signature.parameters

            kwargs = {
                "text": text,
                "audio_prompt_path": effective_audio_prompt,  # None if voice already cached
                "temperature": temperature,
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight,
                "chunk_size": chunk_size,
            }

            if supports_language_id:
                kwargs["language_id"] = language_id
            elif language_id and language_id != "en":
                logger.warning(
                    f"Language '{language_id}' requested but model does not support it in streaming mode."
                )

            for audio_chunk, metrics in chatterbox_model.generate_stream(**kwargs):
                yield audio_chunk, metrics
        else:
            # Fallback: use non-streaming generate() and yield full audio as one chunk
            logger.warning("Model does not support generate_stream, falling back to generate()")
            generate_signature = inspect.signature(chatterbox_model.generate)
            supports_language_id = 'language_id' in generate_signature.parameters

            kwargs = {
                "text": text,
                "audio_prompt_path": effective_audio_prompt,  # None if voice already cached
                "temperature": temperature,
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight,
            }

            if supports_language_id:
                kwargs["language_id"] = language_id

            audio = chatterbox_model.generate(**kwargs)
            yield audio, {}

    except Exception as e:
        logger.error(f"Error during streaming TTS synthesis: {e}", exc_info=True)
        yield None, None

