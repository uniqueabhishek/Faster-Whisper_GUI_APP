"""Audio preprocessing utilities for Faster-Whisper GUI."""

from __future__ import annotations

import logging
import subprocess
import tempfile
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

# Import transcriber to ensure VAD model patches are applied
# This MUST be imported before any faster_whisper.vad imports
try:
    import transcriber  # noqa: F401
    LOGGER = logging.getLogger(__name__)
    LOGGER.info("Transcriber imported - VAD patches applied")
except ImportError:
    LOGGER = logging.getLogger(__name__)
    LOGGER.warning("Could not import transcriber - VAD may not work correctly")


@dataclass
class PreprocessingConfig:
    """Configuration for audio preprocessing operations."""
    convert_to_wav: bool = True
    trim_silence: bool = False
    normalize_audio: bool = False
    reduce_noise: bool = False
    remove_music: bool = False
    target_db: float = -20.0
    output_dir: Optional[Path] = None


def convert_to_wav_16khz_mono(
    input_path: Path,
    output_path: Path,
    cancel_check: Optional[Callable[[], bool]] = None
) -> bool:
    """
    Convert any audio format to 16kHz mono WAV using ffmpeg.

    Args:
        input_path: Path to input audio file
        output_path: Path where converted WAV will be saved
        cancel_check: Optional callable that returns True if cancellation is requested

    Returns:
        True if successful, False otherwise
    """
    try:
        LOGGER.info("Converting %s to 16kHz mono WAV...", input_path.name)

        # ffmpeg -i input -ar 16000 -ac 1 -c:a pcm_s16le output.wav
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(output_path)
        ]

        # Use Popen to allow cancellation
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW
        )

        # Poll for completion while checking for cancellation
        while True:
            if cancel_check and cancel_check():
                process.kill()
                LOGGER.info("Conversion cancelled for %s", input_path.name)
                # Clean up partial file
                if output_path.exists():
                    try:
                        os.unlink(output_path)
                    except Exception:
                        pass
                return False

            if process.poll() is not None:
                break

            import time
            time.sleep(0.1)

        if process.returncode != 0:
            LOGGER.error("ffmpeg conversion failed with return code: %d", process.returncode)
            return False

        LOGGER.info("Successfully converted %s to WAV", input_path.name)
        return True

    except Exception as e:
        LOGGER.error("Conversion failed for %s: %s", input_path.name, str(e))
        return False


def normalize_audio(
    input_path: Path,
    output_path: Path,
    target_db: float = -20.0,
    cancel_check: Optional[Callable[[], bool]] = None
) -> bool:
    """
    Normalize audio volume to target dB using ffmpeg loudnorm filter.

    Args:
        input_path: Path to input audio file
        output_path: Path where normalized audio will be saved
        target_db: Target integrated loudness in LUFS (default: -20.0)
        cancel_check: Optional callable that returns True if cancellation is requested

    Returns:
        True if successful, False otherwise
    """
    try:
        LOGGER.info("Normalizing %s to %s dB...", input_path.name, target_db)

        # ffmpeg loudnorm filter for EBU R128 normalization
        # I = integrated loudness target
        # TP = true peak target
        # LRA = loudness range target
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-af", f"loudnorm=I={target_db}:TP=-1.5:LRA=11",
            str(output_path)
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW
        )

        while True:
            if cancel_check and cancel_check():
                process.kill()
                LOGGER.info("Normalization cancelled for %s", input_path.name)
                if output_path.exists():
                    try:
                        os.unlink(output_path)
                    except Exception:
                        pass
                return False

            if process.poll() is not None:
                break

            import time
            time.sleep(0.1)

        if process.returncode != 0:
            LOGGER.error("ffmpeg normalization failed with return code: %d", process.returncode)
            return False

        LOGGER.info("Successfully normalized %s", input_path.name)
        return True

    except Exception as e:
        LOGGER.error("Normalization failed for %s: %s", input_path.name, str(e))
        return False


def reduce_noise(
    input_path: Path,
    output_path: Path,
    cancel_check: Optional[Callable[[], bool]] = None
) -> bool:
    """
    Apply noise reduction using ffmpeg afftdn filter.

    Args:
        input_path: Path to input audio file
        output_path: Path where denoised audio will be saved
        cancel_check: Optional callable that returns True if cancellation is requested

    Returns:
        True if successful, False otherwise
    """
    try:
        LOGGER.info("Reducing noise in %s...", input_path.name)

        # afftdn = FFT Denoiser
        # nf = noise floor in dB (default: -50, we use -25 for moderate noise reduction)
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-af", "afftdn=nf=-25",
            str(output_path)
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW
        )

        while True:
            if cancel_check and cancel_check():
                process.kill()
                LOGGER.info("Noise reduction cancelled for %s", input_path.name)
                if output_path.exists():
                    try:
                        os.unlink(output_path)
                    except Exception:
                        pass
                return False

            if process.poll() is not None:
                break

            import time
            time.sleep(0.1)

        if process.returncode != 0:
            LOGGER.error("ffmpeg noise reduction failed with return code: %d", process.returncode)
            return False

        LOGGER.info("Successfully reduced noise in %s", input_path.name)
        return True

    except Exception as e:
        LOGGER.error("Noise reduction failed for %s: %s", input_path.name, str(e))
        return False


def remove_music(
    input_path: Path,
    output_path: Path,
    cancel_check: Optional[Callable[[], bool]] = None
) -> bool:
    """
    Remove background music using ffmpeg highpass filter to isolate speech frequencies.

    This uses a combination of filters to suppress music while preserving speech:
    - High-pass filter to remove low-frequency music
    - Band-pass filter to focus on speech frequencies (200Hz-3500Hz)

    Args:
        input_path: Path to input audio file
        output_path: Path where processed audio will be saved
        cancel_check: Optional callable that returns True if cancellation is requested

    Returns:
        True if successful, False otherwise
    """
    try:
        LOGGER.info("Removing background music from %s...", input_path.name)

        # Use a combination of filters to isolate speech frequencies:
        # 1. highpass=f=200: Remove frequencies below 200Hz (removes bass/low music)
        # 2. lowpass=f=3500: Remove frequencies above 3500Hz (human speech range)
        # 3. Optional: bandreject to remove specific music frequency ranges

        # Speech-optimized filter chain
        filter_chain = "highpass=f=200,lowpass=f=3500"

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-af", filter_chain,
            str(output_path)
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW
        )

        while True:
            if cancel_check and cancel_check():
                process.kill()
                LOGGER.info("Music removal cancelled for %s", input_path.name)
                if output_path.exists():
                    try:
                        os.unlink(output_path)
                    except Exception:
                        pass
                return False

            if process.poll() is not None:
                break

            import time
            time.sleep(0.1)

        if process.returncode != 0:
            LOGGER.error("ffmpeg music removal failed with return code: %d", process.returncode)
            return False

        LOGGER.info("Successfully removed music from %s", input_path.name)
        return True

    except Exception as e:
        LOGGER.error("Music removal failed for %s: %s", input_path.name, str(e))
        return False


def trim_silence_vad(
    input_path: Path,
    output_path: Path,
    cancel_check: Optional[Callable[[], bool]] = None
) -> bool:
    """
    Use VAD (Voice Activity Detection) to remove ALL silence segments from audio.
    This concatenates all detected speech segments, removing silence from anywhere in the audio.
    Reuses the existing silero_vad.onnx model.

    Args:
        input_path: Path to input audio file
        output_path: Path where trimmed audio will be saved
        cancel_check: Optional callable that returns True if cancellation is requested

    Returns:
        True if successful, False otherwise
    """
    try:
        LOGGER.info("Trimming silence from %s using VAD...", input_path.name)

        # Import dependencies for VAD
        try:
            import numpy as np
            import soundfile as sf
            import sys

            # Import VAD functions (reuse existing patch from transcriber.py)
            from faster_whisper.vad import get_speech_timestamps, VadOptions

        except ImportError as e:
            LOGGER.error("Missing dependencies for VAD trimming: %s", str(e))
            LOGGER.info("Falling back to simple silence trimming with ffmpeg...")
            return _trim_silence_ffmpeg(input_path, output_path, cancel_check)

        # Load audio
        audio, sample_rate = sf.read(str(input_path))

        if cancel_check and cancel_check():
            return False

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample to 16kHz for VAD if needed
        if sample_rate != 16000:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            except ImportError:
                LOGGER.warning("librosa not available for resampling, using original sample rate")

        if cancel_check and cancel_check():
            return False

        # Configure VAD options as documented in preprocessing_gui.py tooltip
        # - min_silence_duration_ms=3000 (3 seconds) - Only remove silences longer than 3s
        # - speech_pad_ms=1000 (1 second) - Add 1s padding before/after speech
        # - threshold=0.1 (10% confidence) - Very sensitive detection
        vad_options = VadOptions(
            min_silence_duration_ms=3000,
            speech_pad_ms=1000,
            threshold=0.1
        )

        # Get speech timestamps using proper VAD function
        speech_timestamps = get_speech_timestamps(
            audio,
            vad_options=vad_options,
            sampling_rate=sample_rate
        )

        if cancel_check and cancel_check():
            return False

        # If no speech detected, keep the original file
        if not speech_timestamps:
            LOGGER.warning("No speech detected in %s, keeping original", input_path.name)
            # Copy the file instead of trimming
            import shutil
            shutil.copy2(str(input_path), str(output_path))
            return True

        # Extract ALL speech segments and concatenate them
        speech_segments = []

        LOGGER.info("Found %d speech segments in %s", len(speech_timestamps), input_path.name)

        for i, segment in enumerate(speech_timestamps):
            if cancel_check and cancel_check():
                return False

            # get_speech_timestamps returns sample indices, not seconds
            start_sample = segment['start']
            end_sample = segment['end']

            # Extract this speech segment
            speech_chunk = audio[start_sample:end_sample]
            speech_segments.append(speech_chunk)

            LOGGER.debug("Segment %d: %.2fs - %.2fs (duration: %.2fs)",
                        i + 1, start_sample / sample_rate, end_sample / sample_rate,
                        (end_sample - start_sample) / sample_rate)

        # Concatenate all speech segments
        trimmed_audio = np.concatenate(speech_segments)

        if cancel_check and cancel_check():
            return False

        # Save trimmed audio
        sf.write(str(output_path), trimmed_audio, sample_rate)

        duration_removed = (len(audio) - len(trimmed_audio)) / sample_rate
        LOGGER.info("Removed %.2f seconds of silence from %s (kept %d speech segments)",
                   duration_removed, input_path.name, len(speech_timestamps))
        return True

    except Exception as e:
        LOGGER.error("VAD trimming failed for %s: %s", input_path.name, str(e))
        # Fallback to ffmpeg silence trimming
        LOGGER.info("Falling back to ffmpeg silence trimming...")
        return _trim_silence_ffmpeg(input_path, output_path, cancel_check)


def _trim_silence_ffmpeg(
    input_path: Path,
    output_path: Path,
    cancel_check: Optional[Callable[[], bool]] = None
) -> bool:
    """
    Fallback: trim silence using ffmpeg silenceremove filter.

    Args:
        input_path: Path to input audio file
        output_path: Path where trimmed audio will be saved
        cancel_check: Optional callable that returns True if cancellation is requested

    Returns:
        True if successful, False otherwise
    """
    try:
        LOGGER.info("Trimming silence from %s using ffmpeg...", input_path.name)

        # silenceremove filter
        # start_periods=1: remove silence from beginning
        # start_duration=0.5: minimum silence duration to remove (0.5s)
        # start_threshold=-50dB: silence threshold
        # stop_periods=-1: remove silence from end
        # stop_duration=0.5
        # stop_threshold=-50dB
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-af", "silenceremove=start_periods=1:start_duration=0.5:start_threshold=-50dB:stop_periods=-1:stop_duration=0.5:stop_threshold=-50dB",
            str(output_path)
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW
        )

        while True:
            if cancel_check and cancel_check():
                process.kill()
                LOGGER.info("Silence trimming cancelled for %s", input_path.name)
                if output_path.exists():
                    try:
                        os.unlink(output_path)
                    except Exception:
                        pass
                return False

            if process.poll() is not None:
                break

            import time
            time.sleep(0.1)

        if process.returncode != 0:
            stderr = process.stderr.read().decode()
            LOGGER.error("ffmpeg silence trimming failed: %s", stderr)
            return False

        LOGGER.info("Successfully trimmed silence from %s", input_path.name)
        return True

    except Exception as e:
        LOGGER.error("Silence trimming failed for %s: %s", input_path.name, str(e))
        return False


def preprocess_audio(
    input_path: Path,
    config: PreprocessingConfig,
    cancel_check: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[str, int], None]] = None
) -> Optional[Path]:
    """
    Apply all selected preprocessing operations to an audio file.

    Args:
        input_path: Path to input audio file
        config: Preprocessing configuration
        cancel_check: Optional callable that returns True if cancellation is requested
        progress_callback: Optional callable that takes (step_name: str, step_progress: int)
                          to report progress for each preprocessing step

    Returns:
        Path to preprocessed file if successful, None otherwise
    """
    import shutil

    # Check if ffmpeg is available
    if not shutil.which("ffmpeg"):
        LOGGER.error("ffmpeg not found. Cannot preprocess audio.")
        return None

    try:
        # Determine output directory
        if config.output_dir:
            output_dir = config.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Create temp directory
            output_dir = Path(tempfile.mkdtemp(prefix="whisper_preprocess_"))

        # Generate output filename
        output_filename = f"{input_path.stem}_preprocessed.wav"
        final_output = output_dir / output_filename

        # Track current file (starts as input, gets replaced with each step)
        current_file = input_path
        temp_files = []

        # Calculate total enabled steps for progress tracking
        enabled_steps = []
        if config.convert_to_wav:
            enabled_steps.append("Converting to WAV")
        if config.reduce_noise:
            enabled_steps.append("Removing Background Noise")
        if config.remove_music:
            enabled_steps.append("Removing Background Music")
        if config.normalize_audio:
            enabled_steps.append("Normalizing Audio Volume")
        if config.trim_silence:
            enabled_steps.append("Trimming Silence (VAD)")

        total_steps = len(enabled_steps)
        current_step = 0

        # Helper to emit step progress
        def emit_step_progress(step_name: str):
            nonlocal current_step
            if progress_callback:
                # Calculate progress: (current_step / total_steps) * 100
                step_percent = int((current_step / total_steps) * 100) if total_steps > 0 else 0
                progress_callback(step_name, step_percent)
            current_step += 1

        # Step 1: Convert to WAV/16kHz Mono (always do this if selected)
        if config.convert_to_wav:
            emit_step_progress("Converting to WAV")
            temp_wav = output_dir / f"{input_path.stem}_temp_wav.wav"
            if not convert_to_wav_16khz_mono(current_file, temp_wav, cancel_check):
                _cleanup_temp_files(temp_files)
                return None
            temp_files.append(temp_wav)
            current_file = temp_wav

        # Step 2: Reduce Noise (works best on full spectrum before filtering)
        if config.reduce_noise:
            emit_step_progress("Removing Background Noise")
            temp_denoised = output_dir / f"{input_path.stem}_temp_denoised.wav"
            if not reduce_noise(current_file, temp_denoised, cancel_check):
                _cleanup_temp_files(temp_files)
                return None
            temp_files.append(temp_denoised)
            current_file = temp_denoised

        # Step 3: Remove Music (isolate speech frequencies before normalization)
        if config.remove_music:
            emit_step_progress("Removing Background Music")
            temp_no_music = output_dir / f"{input_path.stem}_temp_no_music.wav"
            if not remove_music(current_file, temp_no_music, cancel_check):
                _cleanup_temp_files(temp_files)
                return None
            temp_files.append(temp_no_music)
            current_file = temp_no_music

        # Step 4: Normalize Audio (on speech-band only for accurate loudness targeting)
        if config.normalize_audio:
            emit_step_progress("Normalizing Audio Volume")
            temp_normalized = output_dir / f"{input_path.stem}_temp_normalized.wav"
            if not normalize_audio(current_file, temp_normalized, config.target_db, cancel_check):
                _cleanup_temp_files(temp_files)
                return None
            temp_files.append(temp_normalized)
            current_file = temp_normalized

        # Step 5: Trim Silence with VAD (works best on normalized, clean speech)
        if config.trim_silence:
            emit_step_progress("Trimming Silence (VAD)")
            temp_trimmed = output_dir / f"{input_path.stem}_temp_trimmed.wav"
            if not trim_silence_vad(current_file, temp_trimmed, cancel_check):
                _cleanup_temp_files(temp_files)
                return None
            temp_files.append(temp_trimmed)
            current_file = temp_trimmed

        # Copy final result to output path
        if current_file != final_output:
            shutil.copy2(str(current_file), str(final_output))

        # Cleanup temp files
        _cleanup_temp_files(temp_files)

        LOGGER.info("Preprocessing completed: %s -> %s", input_path.name, final_output.name)
        return final_output

    except Exception as e:
        LOGGER.error("Preprocessing failed for %s: %s", input_path.name, str(e))
        return None


def _cleanup_temp_files(temp_files: list[Path]) -> None:
    """Clean up temporary files."""
    for temp_file in temp_files:
        try:
            if temp_file.exists():
                os.unlink(temp_file)
        except Exception as e:
            LOGGER.warning("Failed to delete temp file %s: %s", temp_file, str(e))
