"""Memory management utilities for Faster-Whisper transcription."""

from __future__ import annotations

import gc
import logging
import os
from typing import TYPE_CHECKING

import psutil

if TYPE_CHECKING:
    from typing import Dict, Optional

LOGGER = logging.getLogger(__name__)


class MemoryManager:
    """Manages memory cleanup and monitoring."""

    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage statistics.

        Returns:
            Dictionary with memory stats in MB.
        """
        try:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()

            return {
                "rss_mb": mem_info.rss / (1024 * 1024),  # Resident Set Size
                "vms_mb": mem_info.vms / (1024 * 1024),  # Virtual Memory Size
                "percent": process.memory_percent(),
            }
        except Exception as e:  # pylint: disable=broad-except
            LOGGER.warning("Failed to get memory usage: %s", e)
            return {"rss_mb": 0, "vms_mb": 0, "percent": 0}

    @staticmethod
    def log_memory_usage(prefix: str = "") -> None:
        """Log current memory usage."""
        stats = MemoryManager.get_memory_usage()
        LOGGER.info(
            "%sMemory: %.0f MB (RSS), %.0f MB (VMS), %.1f%%",
            f"{prefix} " if prefix else "",
            stats["rss_mb"],
            stats["vms_mb"],
            stats["percent"],
        )

    @staticmethod
    def cleanup_memory(aggressive: bool = False) -> None:
        """Trigger garbage collection to free memory.

        Args:
            aggressive: If True, perform multiple GC passes.
        """
        LOGGER.debug("Running garbage collection...")

        if aggressive:
            # Multiple GC passes to clean up circular references
            for generation in range(3):
                collected = gc.collect(generation)
                LOGGER.debug(
                    "GC generation %d: collected %d objects", generation, collected)
        else:
            collected = gc.collect()
            LOGGER.debug("GC collected %d objects", collected)

        # Log memory usage after cleanup
        MemoryManager.log_memory_usage("After cleanup:")

    @staticmethod
    def clear_model_cache(transcriber) -> None:
        """Clear model cache and free GPU/CPU memory.

        Args:
            transcriber: Transcriber instance to clear cache from.
        """
        try:
            if transcriber is None:
                return

            # Check if model exists
            if not hasattr(transcriber, '_model') or transcriber._model is None:  # pylint: disable=protected-access
                return

            LOGGER.info("Clearing model cache...")

            # For faster-whisper models, we can't directly unload
            # but we can try to clear any cached segments
            model = transcriber._model  # pylint: disable=protected-access

            # Clear any cached data in the model
            if hasattr(model, 'feature_extractor'):
                # Feature extractor may have cached mel spectrograms
                del model.feature_extractor
                model.feature_extractor = None

            # If using GPU, clear CUDA cache
            try:
                import torch  # pylint: disable=import-outside-toplevel
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    LOGGER.info("Cleared CUDA cache")
            except ImportError:
                pass  # PyTorch not available

            # Force garbage collection
            MemoryManager.cleanup_memory(aggressive=True)

        except Exception as e:  # pylint: disable=broad-except
            LOGGER.warning("Failed to clear model cache: %s", e)

    @staticmethod
    def check_memory_available(required_mb: float = 1000) -> bool:
        """Check if sufficient memory is available.

        Args:
            required_mb: Required memory in MB.

        Returns:
            True if sufficient memory available.
        """
        try:
            # Get system memory
            mem = psutil.virtual_memory()
            available_mb = mem.available / (1024 * 1024)

            LOGGER.debug(
                "Memory check: %.0f MB available, %.0f MB required",
                available_mb,
                required_mb,
            )

            return available_mb >= required_mb
        except Exception as e:  # pylint: disable=broad-except
            LOGGER.warning("Failed to check memory: %s", e)
            return True  # Assume OK if check fails

    @staticmethod
    def estimate_memory_needed(
        audio_duration_seconds: float,
        compute_type: str = "float32",
        beam_size: int = 5,
    ) -> float:
        """Estimate memory needed for transcription.

        Args:
            audio_duration_seconds: Duration of audio in seconds.
            compute_type: Model precision (int8, float16, float32).
            beam_size: Beam size for decoding.

        Returns:
            Estimated memory in MB.
        """
        # Base memory for model (approximate)
        model_memory = {
            "int8": 200,
            "float16": 400,
            "float32": 800,
        }.get(compute_type, 800)

        # Audio processing memory (approx 1 MB per minute)
        audio_memory = (audio_duration_seconds / 60) * 1

        # Beam search memory (scales with beam size)
        beam_memory = beam_size * 50

        # Add 50% buffer
        total = (model_memory + audio_memory + beam_memory) * 1.5

        LOGGER.debug(
            "Estimated memory: %.0f MB (model: %.0f, audio: %.0f, beam: %.0f)",
            total,
            model_memory,
            audio_memory,
            beam_memory,
        )

        return total

    @staticmethod
    def cleanup_between_files() -> None:
        """Lightweight cleanup between individual file transcriptions."""
        # Run lightweight GC
        gc.collect(0)  # Only collect generation 0 (youngest objects)

    @staticmethod
    def cleanup_between_batches() -> None:
        """Aggressive cleanup between batch transcriptions."""
        LOGGER.info("Performing batch cleanup...")

        # Log memory before cleanup
        MemoryManager.log_memory_usage("Before cleanup:")

        # Aggressive garbage collection
        MemoryManager.cleanup_memory(aggressive=True)

        # Clear CUDA cache if available
        try:
            import torch  # pylint: disable=import-outside-toplevel
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                LOGGER.info("Cleared CUDA cache")
        except ImportError:
            pass

        # Log memory after cleanup
        MemoryManager.log_memory_usage("After cleanup:")


class MemoryMonitor:
    """Context manager for monitoring memory usage during operations."""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_memory: Optional[Dict[str, float]] = None

    def __enter__(self):
        self.start_memory = MemoryManager.get_memory_usage()
        LOGGER.info(
            "[%s] Starting - Memory: %.0f MB",
            self.operation_name,
            self.start_memory["rss_mb"],
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_memory = MemoryManager.get_memory_usage()
        if self.start_memory is None:
            return
        delta = end_memory["rss_mb"] - self.start_memory["rss_mb"]

        LOGGER.info(
            "[%s] Finished - Memory: %.0f MB (Delta %+.0f MB)",
            self.operation_name,
            end_memory["rss_mb"],
            delta,
        )

        # If memory increased significantly, log warning
        if delta > 500:  # More than 500 MB increase
            LOGGER.warning(
                "[%s] Large memory increase detected: %+.0f MB",
                self.operation_name,
                delta,
            )
