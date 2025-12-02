"""Worker threads for Faster-Whisper GUI."""

from __future__ import annotations

import logging
import time
import threading
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from PyQt5.QtCore import QThread, pyqtSignal

from transcriber import Transcriber, TranscriptionResult

LOGGER = logging.getLogger(__name__)

# Global executor for parallel batch
EXECUTOR = ThreadPoolExecutor(max_workers=4)


class SingleFileWorker(QThread):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        transcriber: Transcriber,
        input_path: Path,
        output_path: Optional[Path],
    ) -> None:
        super().__init__()
        self._transcriber = transcriber
        self._input_path = input_path
        self._output_path = output_path
        self._cancel = False

    def request_cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:


<< << << < HEAD
        LOGGER.info("=== WORKER RUN STARTED ===")
        if self._cancel:
            LOGGER.info("Worker cancelled before start")
            self.failed.emit("Cancelled")
            return
        try:
            LOGGER.info("Starting transcription of: %s", self._input_path)
== == == =
        if self._cancel:
            self.failed.emit("Cancelled")
            return
        try:
>>>>>> > 4c9e366 (fixed 7 + issues in your app: )
            result = self._transcriber.transcribe_file(
                self._input_path,
                output_path=self._output_path,
            )
<< << << < HEAD
            LOGGER.info("Transcription completed successfully")
            if self._cancel:
                LOGGER.info("Worker cancelled after transcription")
                self.failed.emit("Cancelled")
                return
            LOGGER.info("Emitting finished signal...")
            self.finished.emit(result)
            LOGGER.info("Finished signal emitted")
        except Exception as exc:  # noqa
            import traceback
            error_msg = f"{str(exc)}\n\nTraceback:\n{traceback.format_exc()}"
            LOGGER.error("Transcription failed: %s", error_msg)
=======
            if self._cancel:
                self.failed.emit("Cancelled")
                return
            self.finished.emit(result)
        except Exception as exc:  # noqa
>>>>>>> 4c9e366 (fixed 7+ issues in your app:)
            self.failed.emit(str(exc))


class BatchWorker(QThread):
    progress = pyqtSignal(int, int)           # processed, total
    speed = pyqtSignal(float, int)            # avg_time, eta_seconds
    file_status = pyqtSignal(str, str)        # filename, status
    finished = pyqtSignal(object)             # list of TranscriptionResult
    failed = pyqtSignal(str)                  # error message

    def __init__(
        self,
        transcriber: Transcriber,
        input_dir: Path,
        output_dir: Path,
        recursive: bool,
    ) -> None:
        super().__init__()
        self._transcriber = transcriber
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._recursive = recursive
        self._cancel = False
        self._model_lock = threading.Lock()   # ensure thread-safe model calls

    def request_cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        # Collect media files
        try:
            media_files = list(
                self._transcriber.iter_media_files(
                    self._input_dir,
                    recursive=self._recursive,
                )
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return

        total = len(media_files)
        if total == 0:
            self.finished.emit([])
            return

        results: List[TranscriptionResult] = []
        start_time = time.time()
        processed = 0

        # Internal job with lock for thread-safety
        def _job(path: Path) -> Optional[TranscriptionResult]:
            if self._cancel:
                return None

            self.file_status.emit(path.name, "Processing")
            try:
                with self._model_lock:
                    result = self._transcriber.transcribe_file(
                        path,
                        output_path=self._output_dir / f"{path.stem}.txt",
                    )
                return result
            except Exception:
                return None

        # Submit all futures to global executor
        futures = {EXECUTOR.submit(_job, p): p for p in media_files}

        try:
            for future in as_completed(futures):
                if self._cancel:
                    self.failed.emit("Cancelled")
                    return

                processed += 1
                media_path = futures[future]

                result = future.result()
                if result is not None:
                    results.append(result)
                    self.file_status.emit(media_path.name, "Done")
                else:
                    self.file_status.emit(media_path.name, "Failed")

                # Speed + ETA calculations
                elapsed = time.time() - start_time
                avg_time = elapsed / processed
                eta = int((total - processed) * avg_time)

                self.progress.emit(processed, total)
                self.speed.emit(avg_time, eta)

        except Exception as exc:
            self.failed.emit(str(exc))
            return

        self.finished.emit(results)
