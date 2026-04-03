# File: stt_engine.py
# Core STT model loading and speech-to-text transcription logic.

import logging
import torch
from typing import Optional
from pathlib import Path

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None

from config import get_stt_device, get_stt_model_size, get_stt_language
from models import TranscriptionResult

logger = logging.getLogger(__name__)


class STTEngine:
    """Speech-to-Text engine using OpenAI Whisper."""
    
    def __init__(self):
        self.model: Optional[whisper.Whisper] = None
        self.model_loaded: bool = False
        self.device: Optional[str] = None


    def _test_device_functionality(self, device: str) -> bool:
        """
        Tests if the specified device is functional for torch operations.
        
        Args:
            device: Device string ('cuda', 'mps', or 'cpu')
            
        Returns:
            bool: True if device works, False otherwise.
        """
        try:
            test_tensor = torch.tensor([1.0])
            test_tensor = test_tensor.to(device)
            test_tensor = test_tensor.cpu()
            return True
        except Exception as e:
            logger.warning(f"{device.upper()} functionality test failed: {e}")
            return False

    def load_model(self) -> bool:
        """
        Loads the Whisper STT model.
        
        Returns:
            bool: True if the model was loaded successfully, False otherwise.
        """
        if not WHISPER_AVAILABLE:
            logger.error("Whisper library not available. Cannot load STT model.")
            return False
        
        if self.model_loaded:
            logger.info("STT model is already loaded.")
            return True
        
        try:
            # Determine processing device
            device_setting = get_stt_device()
            
            if device_setting == "auto":
                if self._test_device_functionality("cuda"):
                    resolved_device_str = "cuda"
                    logger.info("CUDA functionality test passed. Using CUDA for STT.")
                elif self._test_device_functionality("mps"):
                    resolved_device_str = "mps" 
                    logger.info("MPS functionality test passed. Using MPS for STT.")
                else:
                    resolved_device_str = "cpu"
                    logger.info("Using CPU for STT.")
            else:
                resolved_device_str = device_setting
                if not self._test_device_functionality(resolved_device_str):
                    logger.warning(f"Requested device {resolved_device_str} failed test. Falling back to CPU.")
                    resolved_device_str = "cpu"
            
            self.device = resolved_device_str
            model_size = get_stt_model_size()
            
            logger.info(f"Loading Whisper model '{model_size}' on device '{self.device}'...")
            self.model = whisper.load_model(model_size, device=self.device)
            
            self.model_loaded = True
            logger.info(f"STT model '{model_size}' loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load STT model: {e}", exc_info=True)
            self.model = None
            self.model_loaded = False
            return False

    @staticmethod
    def _normalize_language(language: Optional[str]) -> Optional[str]:
        """Normalize language value from request/config to Whisper-compatible value."""
        if language is None:
            return None

        normalized = str(language).strip().lower()
        if normalized in {"", "auto", "none", "null"}:
            return None
        return normalized

    def transcribe_file_with_metadata(
        self, audio_file_path: str, language: Optional[str] = None
    ) -> Optional[dict]:
        """
        Transcribes audio from a file path and returns both text and effective language.

        Args:
            audio_file_path: Path to the audio file
            language: Language code or None for auto-detection

        Returns:
            Dict with text/language metadata or None if transcription fails
        """
        if not self.model_loaded or self.model is None:
            logger.error("STT model is not loaded. Cannot transcribe audio.")
            return None

        try:
            audio_path = Path(audio_file_path)
            if not audio_path.exists():
                logger.error(f"Audio file not found: {audio_file_path}")
                return None

            configured_language = get_stt_language()
            effective_language = self._normalize_language(language)
            if effective_language is None:
                effective_language = self._normalize_language(configured_language)

            logger.info(
                "Transcribing audio file: %s | language=%s",
                audio_file_path,
                effective_language if effective_language else "auto",
            )

            result = self.model.transcribe(str(audio_path), language=effective_language)
            transcribed_text = str(result.get("text", "")).strip()
            detected_language = result.get("language")
            resolved_language = detected_language or effective_language or "auto"

            logger.info(
                "Transcription completed. Length: %d characters | resolved_language=%s",
                len(transcribed_text),
                resolved_language,
            )
            return {
                "text": transcribed_text,
                "language": resolved_language,
                "detected_language": detected_language,
                "requested_language": effective_language,
            }

        except Exception as e:
            logger.error(f"Error during transcription: {e}", exc_info=True)
            return None

    def transcribe_file(self, audio_file_path: str, language: Optional[str] = None) -> Optional[str]:
        """
        Transcribes audio from a file path.
        
        Args:
            audio_file_path: Path to the audio file
            language: Language code or None for auto-detection
            
        Returns:
            Transcribed text or None if transcription fails
        """
        metadata = self.transcribe_file_with_metadata(audio_file_path, language)
        if metadata is None:
            return None
        return metadata.get("text")

    def transcribe_numpy(self, audio_array: 'np.ndarray', language: Optional[str] = None) -> Optional[str]:
        """
        Transcribes audio from a numpy array directly.
        
        Args:
            audio_array: Float32 numpy array with audio data (mono, any sample rate)
            language: Language code or None for auto-detection
            
        Returns:
            Transcribed text or None if transcription fails
        """
        if not self.model_loaded or self.model is None:
            logger.error("STT model is not loaded. Cannot transcribe audio.")
            return None
        
        try:
            import numpy as np
            
            if not isinstance(audio_array, np.ndarray):
                logger.error("Audio input must be a numpy array")
                return None
            
            if len(audio_array) == 0:
                logger.warning("Empty audio array provided")
                return None
            
            # Use requested/configured language or auto-detection
            detect_language = self._normalize_language(language)
            if detect_language is None:
                detect_language = self._normalize_language(get_stt_language())
            language_param = detect_language
            
            logger.debug(f"Transcribing numpy array: shape={audio_array.shape}, dtype={audio_array.dtype}")
            result = self.model.transcribe(audio_array, language=language_param)
            
            transcribed_text = result["text"].strip()
            logger.debug(f"Numpy transcription completed. Length: {len(transcribed_text)} characters")
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Error during numpy transcription: {e}", exc_info=True)
            return None

    def transcribe_numpy_with_timing(self, audio_array: 'np.ndarray', language: Optional[str] = None) -> Optional[TranscriptionResult]:
        """
        Transcribes audio from a numpy array and returns full result with timing information.
        
        Args:
            audio_array: Float32 numpy array with audio data (mono, any sample rate)
            language: Language code or None for auto-detection
            
        Returns:
            Full transcription result dict with timing information, or None if transcription fails
        """
        if not self.model_loaded or self.model is None:
            logger.error("STT model is not loaded. Cannot transcribe audio.")
            return None
        
        try:
            import numpy as np
            
            if not isinstance(audio_array, np.ndarray):
                logger.error("Audio input must be a numpy array")
                return None
            
            if len(audio_array) == 0:
                logger.warning("Empty audio array provided")
                return None
            
            # Use requested/configured language or auto-detection
            detect_language = self._normalize_language(language)
            if detect_language is None:
                detect_language = self._normalize_language(get_stt_language())
            language_param = detect_language
            
            logger.debug(f"Transcribing numpy array with timing: shape={audio_array.shape}, dtype={audio_array.dtype}")
            raw_result = self.model.transcribe(audio_array, language=language_param)
            
            logger.debug(f"Numpy transcription with timing completed. Segments: {len(raw_result.get('segments', []))}")
            
            # Convert raw result to our typed model
            try:
                transcription_result = TranscriptionResult(**raw_result)
                return transcription_result
            except Exception as e:
                logger.error(f"Error converting transcription result to typed model: {e}")
                logger.debug(f"Raw result keys: {list(raw_result.keys()) if raw_result else 'None'}")
                return None
            
        except Exception as e:
            logger.error(f"Error during numpy transcription with timing: {e}", exc_info=True)
            return None