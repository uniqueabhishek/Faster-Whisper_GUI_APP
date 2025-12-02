"""Transcription utilities for Faster-Whisper GUI app."""

from __future__ import annotations
from faster_whisper import WhisperModel

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

<< << << < HEAD
try:
    from faster_whisper import WhisperModel
    LOGGER = logging.getLogger(__name__)
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
== == == =

LOGGER = logging.getLogger(__name__)
>>>>>> > 4c9e366 (fixed 7 + issues in your app: )


@dataclass(frozen=True)
class TranscriptionConfig:
    model_name: str
    device: str = "auto"


<< << << < HEAD
    compute_type: str = "default"
== == == =
    compute_type: str = "int8_float16"
>>>>>> > 4c9e366 (fixed 7 + issues in your app: )
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


<< << << < HEAD
        LOGGER.info("Model path used: %s", self._config.model_name)
== == == =
        print("MODEL PATH USED:", self._config.model_name)
>>>>>> > 4c9e366 (fixed 7 + issues in your app: )
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
<< << << < HEAD
        LOGGER.info("Device: %s, Compute type: %s",
                    self._config.device, self._config.compute_type)

        try:
            # Try with explicit cpu device first as it's most compatible
            LOGGER.info("Attempting to load model with CPU device...")
            model = WhisperModel(
                str(model_path),
                device="cpu",
                compute_type="int8",
            )
            LOGGER.info("Model loaded successfully with CPU/int8")
            return model
        except Exception as e:
            LOGGER.error("Failed to load with CPU/int8: %s", str(e))
            LOGGER.info("Trying with default settings...")
            try:
                model = WhisperModel(
                    str(model_path),
                    device="cpu",
                )
                LOGGER.info("Model loaded successfully with CPU/default")
                return model
            except Exception as e2:
                LOGGER.error("Failed to load with CPU/default: %s", str(e2))
                raise RuntimeError(
                    f"Failed to load model. Errors:\n1. {str(e)}\n2. {str(e2)}")
== == == =

        return WhisperModel(
            str(model_path),
            device=self._config.device,
            compute_type=self._config.compute_type,
            download=False,
        )
>>>>>> > 4c9e366 (fixed 7 + issues in your app: )

    def transcribe_file(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
    ) -> TranscriptionResult:
<< << << < HEAD
        LOGGER.info("=== TRANSCRIBE_FILE CALLED ===")
        LOGGER.info("Input: %s", input_path)
        LOGGER.info("Output: %s", output_path)

        if not input_path.is_file():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        LOGGER.info("Calling model.transcribe()...")
== == == =
        if not input_path.is_file():
            raise FileNotFoundError(f"Input file not found: {input_path}")

>>>>>> > 4c9e366 (fixed 7 + issues in your app: )
        segments, info = self._model.transcribe(
            str(input_path),
            language=self._config.language,
            beam_size=self._config.beam_size,
            best_of=self._config.best_of,
            vad_filter=True,
        )
<< << << < HEAD
        LOGGER.info("Model.transcribe() completed")

        LOGGER.info("Processing segments...")
== == == =

>>>>>> > 4c9e366 (fixed 7 + issues in your app: )
        lines: List[str] = [
            segment.text.strip()
            for segment in segments
            if getattr(segment, "text", "").strip()
        ]
        text = "\n".join(lines)
<< << << < HEAD
        LOGGER.info("Processed %d text lines", len(lines))

        resolved_output: Optional[Path] = None
        if output_path is not None:
            LOGGER.info("Saving to file...")
== == == =

        resolved_output: Optional[Path] = None
        if output_path is not None:
>>>>>> > 4c9e366 (fixed 7 + issues in your app: )
            resolved_output = output_path
            resolved_output.parent.mkdir(parents=True, exist_ok=True)
            resolved_output.write_text(text, encoding="utf-8")
            LOGGER.info("Saved transcription to %s", resolved_output)

<< << << < HEAD
        LOGGER.info("Creating result object...")
== == == =
>>>>>> > 4c9e366 (fixed 7 + issues in your app: )
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
