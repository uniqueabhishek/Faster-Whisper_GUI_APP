"""Transcription utilities for Faster-Whisper GUI app."""

from __future__ import annotations
from faster_whisper import WhisperModel

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional

try:
    from faster_whisper import WhisperModel
    import faster_whisper.vad
    import faster_whisper.transcribe

    LOGGER = logging.getLogger(__name__)

    import sys
    import os

    # Monkey patch VAD model path for offline use
    if getattr(sys, 'frozen', False):
        # If running as a bundled exe, look in the temporary folder
        _VAD_PATH = Path(sys._MEIPASS) / "assets" / "silero_vad.onnx"
    else:
        _VAD_PATH = Path(__file__).parent / "assets" / "silero_vad.onnx"

    LOGGER.info("Checking VAD path for patch: %s", _VAD_PATH)

    if _VAD_PATH.exists():
        import numpy as np
        class SessionWrapper:
            def __init__(self, session):
                self._session = session
                self._inputs = [i.name for i in session.get_inputs()]
                LOGGER.info("VAD Model inputs: %s", self._inputs)

            def run(self, output_names, input_feed, run_options=None):
                # Get input audio and batch size
                audio_input = input_feed.get('input')
                if audio_input is None:
                    return self._session.run(output_names, input_feed, run_options)

                batch_size = audio_input.shape[0]

                # 1. Inject sr if missing and required
                if 'sr' in self._inputs and 'sr' not in input_feed:
                    input_feed['sr'] = np.array([16000], dtype=np.int64)

                # 2. Prepare h/c with correct shape (2, batch_size, 64)
                # We use zeros because faster_whisper uses context-based batching
                # and likely expects stateless processing for these chunks.
                if 'h' in self._inputs:
                    input_feed['h'] = np.zeros((2, batch_size, 64), dtype=np.float32)
                if 'c' in self._inputs:
                    input_feed['c'] = np.zeros((2, batch_size, 64), dtype=np.float32)

                # 3. Run session
                outputs = self._session.run(output_names, input_feed, run_options)

                # 4. Return dummy state to satisfy faster_whisper loop
                # It expects (1, 1, 128) to pass to the next iteration (which we will ignore/reset anyway)
                if len(outputs) == 3:
                    prob, _, _ = outputs
                    dummy_state = np.zeros((1, 1, 128), dtype=np.float32)
                    return [prob, dummy_state, dummy_state]

                return outputs

            def get_inputs(self):
                return self._session.get_inputs()

            def get_outputs(self):
                return self._session.get_outputs()

        def _get_local_vad_model():
            LOGGER.info("Instantiating local VAD model from: %s", _VAD_PATH)
            model = faster_whisper.vad.SileroVADModel(str(_VAD_PATH))
            # Wrap the session to inject 'sr' if needed
            model.session = SessionWrapper(model.session)
            return model

        # Patch faster_whisper.vad.get_vad_model
        if hasattr(faster_whisper.vad, "get_vad_model"):
            faster_whisper.vad.get_vad_model = _get_local_vad_model
            LOGGER.info("Patched faster_whisper.vad.get_vad_model")

        # Patch faster_whisper.transcribe.get_vad_model (crucial if it was imported directly)
        if hasattr(faster_whisper.transcribe, "get_vad_model"):
            faster_whisper.transcribe.get_vad_model = _get_local_vad_model
            LOGGER.info("Patched faster_whisper.transcribe.get_vad_model")

    else:
        LOGGER.warning("VAD model not found at %s. Patch skipped.", _VAD_PATH)

    LOGGER.info("faster_whisper imported successfully")
except ImportError as e:
    LOGGER = logging.getLogger(__name__)
    LOGGER.error("Failed to import faster_whisper: %s", str(e))
    raise

# Check for CTranslate2
try:
    import ctranslate2
    LOGGER.info("ctranslate2 version: %s", ctranslate2.__version__)
except ImportError:
    LOGGER.warning("ctranslate2 not found - this may cause issues")


@dataclass(frozen=True)
class TranscriptionConfig:
    model_name: str
    device: str = "cpu"
    compute_type: str = "int8"
    language: Optional[str] = None
    beam_size: int = 5
    best_of: int = 5


@dataclass(frozen=True)
class TranscriptionResult:
    input_path: Path
    output_path: Optional[Path]
    text: str
    duration_seconds: float


AUDIO_VIDEO_EXTS: tuple[str, ...] = (
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".ogg",
    ".mp4",
    ".mkv",
    ".webm",
)


class Transcriber:
    """High-level wrapper around Faster-Whisper."""

    def __init__(self, config: TranscriptionConfig) -> None:
        self._config = config
        self._model: WhisperModel = self._load_model()

    def _load_model(self) -> WhisperModel:
        LOGGER.info("Model path used: %s", self._config.model_name)
        model_path = Path(self._config.model_name)

        if not model_path.exists() or not model_path.is_dir():
            raise ValueError(
                "Wrong model selected. This file is not a Whisper model."
            )

        expected_files = ["model.bin", "model.int8.bin", "config.json"]
        if not any((model_path / f).exists() for f in expected_files):
            raise ValueError(
                "Wrong model selected. This file is not a Whisper model."
            )

        LOGGER.info("Loading local offline model: %s", model_path)
        LOGGER.info("Device: %s, Compute type: %s",
                    self._config.device, self._config.compute_type)

        try:
            return WhisperModel(
                str(model_path),
                device=self._config.device,
                compute_type=self._config.compute_type,
            )
        except Exception as e:
            LOGGER.error("Failed to load model: %s", str(e))
            raise

    def transcribe_file(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
        beam_size: Optional[int] = None,
        vad_filter: bool = False,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> TranscriptionResult:
        LOGGER.info("=== TRANSCRIBE_FILE CALLED ===")
        LOGGER.info("Input: %s", input_path)
        LOGGER.info("Output: %s", output_path)

        if not input_path.is_file():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        bs = beam_size if beam_size is not None else self._config.beam_size
        lang = language if language else self._config.language

        LOGGER.info("Calling model.transcribe() with beam_size=%d, vad_filter=%s, language=%s, prompt=%s...",
                    bs, vad_filter, lang, initial_prompt)
        # Conservative VAD parameters to prevent cutting text
        vad_params = dict(
            min_silence_duration_ms=1000,
            speech_pad_ms=400,
            threshold=0.35,  # Lower threshold = more sensitive to speech
        ) if vad_filter else None

        try:
            segments, info = self._model.transcribe(
                str(input_path),
                language=lang,
                beam_size=bs,
                best_of=self._config.best_of,
                vad_filter=vad_filter,
                vad_parameters=vad_params,
                initial_prompt=initial_prompt,
            )
        except Exception as e:
            # Fallback for VAD errors (e.g. invalid model file or missing dependencies)
            if vad_filter and ("ONNXRuntimeError" in str(e) or "INVALID_PROTOBUF" in str(e)):
                LOGGER.warning(f"VAD failed to load ({e}). Retrying with VAD disabled.")
                segments, info = self._model.transcribe(
                    str(input_path),
                    language=lang,
                    beam_size=bs,
                    best_of=self._config.best_of,
                    vad_filter=False,
                    initial_prompt=initial_prompt,
                )
            else:
                raise e
        LOGGER.info("Model.transcribe() completed")

        total_duration = info.duration
        LOGGER.info("Audio duration: %.2fs", total_duration)

        LOGGER.info("Processing segments...")
        lines: List[str] = []

        for segment in segments:
            if getattr(segment, "text", "").strip():
                lines.append(segment.text.strip())

            # Update progress
            if progress_callback and total_duration > 0:
                current_time = segment.end
                percent = int((current_time / total_duration) * 100)
                progress_callback(min(percent, 100))

        text = "\n".join(lines)
        LOGGER.info("Processed %d text lines", len(lines))

        resolved_output: Optional[Path] = None
        if output_path is not None:
            LOGGER.info("Saving to file...")
            resolved_output = output_path
            resolved_output.parent.mkdir(parents=True, exist_ok=True)
            resolved_output.write_text(text, encoding="utf-8")
            LOGGER.info("Saved transcription to %s", resolved_output)

        LOGGER.info("Creating result object...")
        return TranscriptionResult(
            input_path=input_path,
            output_path=resolved_output,
            text=text,
            duration_seconds=float(getattr(info, "duration", 0.0)),
        )

    def iter_media_files(
        self, root: Path, recursive: bool = True
    ) -> Iterable[Path]:
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: {root}")

        iterator = root.rglob("*") if recursive else root.iterdir()

        for path in iterator:
            if path.is_file() and path.suffix.lower() in AUDIO_VIDEO_EXTS:
                yield path

    def transcribe_folder(
        self,
        input_dir: Path,
        output_dir: Path,
        recursive: bool = True,
    ) -> List[TranscriptionResult]:
        output_dir.mkdir(parents=True, exist_ok=True)
        results: List[TranscriptionResult] = []

        for media_path in self.iter_media_files(input_dir, recursive):
            output_path = output_dir / f"{media_path.stem}.txt"
            try:
                result = self.transcribe_file(media_path, output_path)
            except FileNotFoundError:
                continue
            results.append(result)

        return results
