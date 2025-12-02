"""PyQt5 GUI for Faster-Whisper transcription app."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QProgressBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QListWidget,
    QStatusBar,
)

from transcriber import (
    TranscriptionConfig,
    Transcriber,
    TranscriptionResult,
)
from workers import BatchWorker, SingleFileWorker

LOGGER = logging.getLogger(__name__)

MEDIA_FILTER = (
    "Media Files (*.mp3 *.wav *.m4a *.flac *.ogg *.mp4 *.mkv *.webm);;"
    "All Files (*)"
)
DEFAULT_WIDTH = 1000
DEFAULT_HEIGHT = 700


class MainWindow(QMainWindow):
    """Main Faster-Whisper GUI window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Faster-Whisper GUI")

        self._transcriber: Optional[Transcriber] = None
        self._single_worker: Optional[SingleFileWorker] = None
        self._batch_worker: Optional[BatchWorker] = None

        self.model_edit: QLineEdit
        self.model_btn: QPushButton
        self.file_edit: QLineEdit
        self.single_output_btn: QPushButton
        self.folder_edit: QLineEdit
        self.output_folder_edit: QLineEdit
        self.recursive_check: QCheckBox
        self.batch_btn: QPushButton
        self.cancel_btn: QPushButton
        self.progress_bar: QProgressBar
        self.output_edit: QTextEdit
        self.file_status_list: QListWidget

        self._build_ui()
        self._create_status_bar()

        # Explicit typing so Pylance knows it's not None
        self._bar: QStatusBar = self.statusBar()

        self.resize(DEFAULT_WIDTH, DEFAULT_HEIGHT)
        self.setMinimumSize(800, 600)

    def show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)

    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)

        self._build_model_row(layout)
        self._build_single_file_row(layout)
        self._build_batch_row(layout)
        self._build_output_area(layout)

        self.setCentralWidget(central)

    def _build_model_row(self, layout: QVBoxLayout) -> None:
        row = QHBoxLayout()

        row.addWidget(QLabel("Model (.bin):"))
        self.model_edit = QLineEdit()
        self.model_edit.setReadOnly(True)
        self.model_edit.setEnabled(False)
        row.addWidget(self.model_edit)

        self.model_btn = QPushButton("Select Model")
<<<<<<< HEAD
        self.model_btn.setEnabled(True)
=======
        self.model_btn.setEnabled(False)
>>>>>>> 4c9e366 (fixed 7+ issues in your app:)
        self.model_btn.clicked.connect(self.on_select_model_clicked)
        row.addWidget(self.model_btn)

        layout.addLayout(row)

    def _build_single_file_row(self, layout: QVBoxLayout) -> None:
        row = QHBoxLayout()

        row.addWidget(QLabel("File:"))
        self.file_edit = QLineEdit()
        row.addWidget(self.file_edit)

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.on_browse_file_clicked)
        row.addWidget(browse_btn)

        self.single_output_btn = QPushButton("Transcribe File")
        self.single_output_btn.setEnabled(False)
        self.single_output_btn.clicked.connect(self.on_transcribe_file_clicked)
        row.addWidget(self.single_output_btn)

        layout.addLayout(row)

    def _build_batch_row(self, layout: QVBoxLayout) -> None:
        input_row = QHBoxLayout()
        input_row.addWidget(QLabel("Folder:"))
        self.folder_edit = QLineEdit()
        input_row.addWidget(self.folder_edit)

        folder_btn = QPushButton("Browse Folder")
        folder_btn.clicked.connect(self.on_browse_folder_clicked)
        input_row.addWidget(folder_btn)
        layout.addLayout(input_row)

        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output Folder:"))
        self.output_folder_edit = QLineEdit()
        output_row.addWidget(self.output_folder_edit)

        output_btn = QPushButton("Choose Output")
        output_btn.clicked.connect(self.on_browse_output_folder_clicked)
        output_row.addWidget(output_btn)

        self.recursive_check = QCheckBox("Include subfolders")
        self.recursive_check.setChecked(True)
        output_row.addWidget(self.recursive_check)

        self.batch_btn = QPushButton("Batch Transcribe")
        self.batch_btn.setEnabled(False)
        self.batch_btn.clicked.connect(self.on_batch_transcribe_clicked)
        output_row.addWidget(self.batch_btn)

        layout.addLayout(output_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFormat("%p%")
        layout.addWidget(self.progress_bar)

        self.file_status_list = QListWidget()
        layout.addWidget(self.file_status_list)

        self.cancel_btn = QPushButton("Cancel Job")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.on_cancel_clicked)
        layout.addWidget(self.cancel_btn)

    def _build_output_area(self, layout: QVBoxLayout) -> None:
        self.output_edit = QTextEdit()
        self.output_edit.setReadOnly(True)
        layout.addWidget(self.output_edit)

    def _create_status_bar(self) -> None:
        bar = self.statusBar()
<<<<<<< HEAD
        if bar:
            bar.showMessage("Ready")
=======
        bar.showMessage("Ready")
>>>>>>> 4c9e366 (fixed 7+ issues in your app:)

    # ---------------------------------------------------------
    # MODEL SELECTION — FIXED + ZERO ERROR
    # ---------------------------------------------------------

    def on_select_model_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Whisper model (*.bin)",
            "",
            "Whisper Model (*.bin);;All Files (*)",
        )
        if not path:
            return

        p = Path(path).resolve()
        folder = p if p.is_dir() else p.parent

        # If model is inside nested folder "model"
        if (folder / "model").is_dir():
            folder = folder / "model"

        model_dir = folder

<<<<<<< HEAD
        # Check for config.json (required)
        if not (model_dir / "config.json").exists():
            self.show_error(
                "Wrong model selected.\nFolder must contain:\n"
                "config.json and either model.bin or model.int8.bin"
            )
            return

        # Check for at least one model file (model.bin or model.int8.bin)
        optional_model_files = ["model.bin", "model.int8.bin"]
        if not any((model_dir / f).exists() for f in optional_model_files):
            self.show_error(
                "Wrong model selected.\nFolder must contain:\n"
                "config.json and either model.bin or model.int8.bin"
=======
        required = ["model.bin", "config.json", "tokenizer.json"]
        if not all((model_dir / f).exists() for f in required):
            self.show_error(
                "Wrong model selected.\nFolder must contain:\n"
                "model.bin, config.json, tokenizer.json"
>>>>>>> 4c9e366 (fixed 7+ issues in your app:)
            )
            return

        self.model_edit.setText(str(model_dir))
        self.single_output_btn.setEnabled(True)
        self.batch_btn.setEnabled(True)

<<<<<<< HEAD
        LOGGER.info("Model directory selected: %s", model_dir)
=======
        print("MODEL DIR SELECTED:", model_dir)
>>>>>>> 4c9e366 (fixed 7+ issues in your app:)

    # ---------------------------------------------------------
    # LAZY MODEL LOAD
    # ---------------------------------------------------------
    def _lazy_load_model(self) -> bool:
        if self._transcriber is not None:
            return True

        self._bar.showMessage("Loading model...")
        try:
            config = TranscriptionConfig(
                model_name=self.model_edit.text().strip(),
                language=None,
            )
            self._transcriber = Transcriber(config)
        except Exception as exc:
<<<<<<< HEAD
            error_msg = str(exc)
            detailed_error = f"Failed to load model:\n\n{error_msg}\n\n"
            detailed_error += "Common solutions:\n"
            detailed_error += "1. Check if faster-whisper is installed correctly\n"
            detailed_error += "2. Ensure you have enough disk space (need 1-2GB free)\n"
            detailed_error += "3. Try: pip install faster-whisper --upgrade\n"
            detailed_error += "4. Check if model files are compatible with your faster-whisper version\n"
            detailed_error += "5. See whisper_gui_debug.log for details"

            LOGGER.exception("Model loading failed")
            self.show_error(detailed_error)
=======
            self.show_error(str(exc))
>>>>>>> 4c9e366 (fixed 7+ issues in your app:)
            self._bar.showMessage("Model load failed.")
            return False

        self.single_output_btn.setEnabled(True)
        self.batch_btn.setEnabled(True)
        self._bar.showMessage("Model loaded.")
        return True

    # ---------------------------------------------------------
    # SINGLE FILE MODE
    # ---------------------------------------------------------
    def on_browse_file_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select media file", "", MEDIA_FILTER
        )
        if path:
            self.file_edit.setText(path)
<<<<<<< HEAD

    def on_transcribe_file_clicked(self) -> None:
        try:
            LOGGER.info("=== TRANSCRIBE CLICKED ===")
            LOGGER.info("Step 1: Loading model...")
            if not self._lazy_load_model():
                LOGGER.error("Model loading failed")
                return

            LOGGER.info("Step 2: Checking transcriber...")
            if self._transcriber is None:
                LOGGER.error("Transcriber is None")
                return

            LOGGER.info("Step 3: Getting input path...")
            input_path = Path(self.file_edit.text().strip())
            LOGGER.info("Input path: %s", input_path)

            if not input_path.is_file():
                self.show_error("Invalid file.")
                return

            LOGGER.info("Step 4: Setting output path...")
            output_path = input_path.with_suffix(".txt")
            LOGGER.info("Output path: %s", output_path)

            LOGGER.info("Step 5: Clearing UI...")
            self.output_edit.clear()
            self.file_status_list.clear()
            self.progress_bar.setValue(0)
            self._bar.showMessage("Transcribing...")

            LOGGER.info("Step 6: Setting busy state...")
            self._set_busy(True)

            LOGGER.info("Step 7: Creating worker thread...")
            worker = SingleFileWorker(
                self._transcriber, input_path, output_path
            )
            self._single_worker = worker

            LOGGER.info("Step 8: Connecting signals...")
            worker.finished.connect(self.on_single_finished)
            worker.failed.connect(self.on_single_failed)

            LOGGER.info("Step 9: Starting worker thread...")
            worker.start()
            LOGGER.info("Worker thread started successfully")

        except Exception as exc:
            LOGGER.exception("Failed to start transcription")
            self.show_error(f"Failed to start transcription:\n{str(exc)}")
            self._set_busy(False)
=======
            self.model_btn.setEnabled(True)

    def on_transcribe_file_clicked(self) -> None:
        if not self._lazy_load_model():
            return

        if self._transcriber is None:
            return

        input_path = Path(self.file_edit.text().strip())
        if not input_path.is_file():
            self.show_error("Invalid file.")
            return

        output_path = input_path.with_suffix(".txt")
        self.output_edit.clear()
        self.file_status_list.clear()
        self.progress_bar.setValue(0)
        self._bar.showMessage("Transcribing...")

        self._set_busy(True)

        worker = SingleFileWorker(
            self._transcriber, input_path, output_path
        )
        self._single_worker = worker
        worker.finished.connect(self.on_single_finished)
        worker.failed.connect(self.on_single_failed)
        worker.start()
>>>>>>> 4c9e366 (fixed 7+ issues in your app:)

    def on_single_finished(self, result: TranscriptionResult) -> None:
        self.output_edit.setPlainText(result.text)
        self._bar.showMessage("Done.")
        self.progress_bar.setValue(100)
        self._single_worker = None
        self._set_busy(False)

    def on_single_failed(self, message: str) -> None:
        self.show_error(message)
        self._bar.showMessage("Error.")
        self._single_worker = None
        self._set_busy(False)

    # ---------------------------------------------------------
    # BATCH MODE
    # ---------------------------------------------------------
    def on_browse_folder_clicked(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select input folder", ""
        )
        if folder:
            self.folder_edit.setText(folder)
<<<<<<< HEAD
=======
            self.model_btn.setEnabled(True)
>>>>>>> 4c9e366 (fixed 7+ issues in your app:)

    def on_browse_output_folder_clicked(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select output folder", ""
        )
        if folder:
            self.output_folder_edit.setText(folder)

    def on_batch_transcribe_clicked(self) -> None:
        if not self._lazy_load_model():
            return

        if self._transcriber is None:
            return

        input_dir = Path(self.folder_edit.text().strip())
        if not input_dir.is_dir():
            self.show_error("Invalid input folder.")
            return

        out_text = self.output_folder_edit.text().strip()
        output_dir = (
            Path(out_text) if out_text else input_dir / "transcripts"
        )

        recursive = self.recursive_check.isChecked()

        self.output_edit.clear()
        self.file_status_list.clear()
        self.progress_bar.setValue(0)
        self._bar.showMessage("Batch running...")
        self._set_busy(True)

        worker = BatchWorker(
            self._transcriber,
            input_dir=input_dir,
            output_dir=output_dir,
            recursive=recursive,
        )
        self._batch_worker = worker

        worker.progress.connect(self.on_batch_progress)
        worker.speed.connect(self.on_speed_update)
        worker.file_status.connect(self.on_file_status_update)
        worker.finished.connect(self.on_batch_finished)
        worker.failed.connect(self.on_batch_failed)
        worker.start()

    def on_file_status_update(self, filename: str, status: str) -> None:
        self.file_status_list.addItem(f"{filename} → {status}")
        self.file_status_list.scrollToBottom()

    def on_batch_progress(self, processed: int, total: int) -> None:
<<<<<<< HEAD
        percent = int(processed * 100 / total) if total > 0 else 0
=======
        percent = int(processed * 100 / total)
>>>>>>> 4c9e366 (fixed 7+ issues in your app:)
        self.progress_bar.setValue(percent)

    def on_speed_update(self, avg_time: float, eta_seconds: int) -> None:
        minutes, seconds = divmod(eta_seconds, 60)
        eta_txt = (
            f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
        )
        self.progress_bar.setFormat(
            f"%p%   •   {avg_time:.2f}s/file   •   ETA {eta_txt}"
        )

    def on_batch_finished(self, results: List[TranscriptionResult]) -> None:
        self._bar.showMessage(f"Completed. Files: {len(results)}")
        self._batch_worker = None
        self._set_busy(False)

    def on_batch_failed(self, message: str) -> None:
        self.show_error(message)
        self._bar.showMessage("Failed.")
        self._batch_worker = None
        self._set_busy(False)

    # ---------------------------------------------------------
    # CANCEL + BUSY
    # ---------------------------------------------------------
    def on_cancel_clicked(self) -> None:
        if self._single_worker:
            self._single_worker.request_cancel()
        if self._batch_worker:
            self._batch_worker.request_cancel()
        self._bar.showMessage("Cancelling...")

    def _set_busy(self, busy: bool) -> None:
        if busy:
<<<<<<< HEAD
            self.setCursor(QCursor(Qt.WaitCursor))  # type: ignore[attr-defined]
=======
            self.setCursor(QCursor(Qt.WaitCursor))
>>>>>>> 4c9e366 (fixed 7+ issues in your app:)
        else:
            self.unsetCursor()

        has_model = self._transcriber is not None

        self.model_btn.setEnabled(not busy)
        self.single_output_btn.setEnabled(not busy and has_model)
        self.batch_btn.setEnabled(not busy and has_model)
        self.cancel_btn.setEnabled(busy)
