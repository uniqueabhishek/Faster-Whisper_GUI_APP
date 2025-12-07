"""Preprocessing window for Faster-Whisper GUI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional
import tempfile

from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGroupBox,
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
    QSlider,
    QAbstractItemView,
    QListWidgetItem,
)

from styles import DARK_THEME_QSS, apply_dark_title_bar
from audio_processor import PreprocessingConfig
from preprocessing_worker import PreprocessingWorker

# Import DragDropWidget and other utilities from gui
# We'll need to import from gui after modifying it
# For now, let's create a simple version here
from PyQt5.QtCore import pyqtSignal, QObject, QUrl
from PyQt5.QtGui import QDragEnterEvent, QDropEvent
from PyQt5.QtWidgets import QFrame

LOGGER = logging.getLogger(__name__)

MEDIA_FILTER = (
    "Media Files (*.mp3 *.wav *.m4a *.flac *.ogg *.mp4 *.mkv *.webm);;"
    "All Files (*)"
)
DEFAULT_WIDTH = 900
DEFAULT_HEIGHT = 700


class LogSignal(QObject):
    """Signal emitter for logging."""
    log_signal = pyqtSignal(str)


class QtLogHandler(logging.Handler):
    """Custom logging handler that emits signals to a QTextEdit."""

    def __init__(self, text_widget: QTextEdit):
        super().__init__()
        self.text_widget = text_widget
        self.emitter = LogSignal()
        self.emitter.log_signal.connect(self._append_text)
        self.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))

    def emit(self, record):
        msg = self.format(record)
        self.emitter.log_signal.emit(msg)

    def _append_text(self, msg: str):
        self.text_widget.append(msg)
        self.text_widget.verticalScrollBar().setValue(
            self.text_widget.verticalScrollBar().maximum()
        )


class DragDropWidget(QFrame):
    """A styled frame that accepts file drops."""

    filesDropped = pyqtSignal(list)

    def __init__(self, title: str = "Drag & Drop Files Here"):
        super().__init__()
        self.setAcceptDrops(True)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #4b5563;
                border-radius: 8px;
                background-color: #262626;
            }
            QFrame:hover {
                border-color: #3b82f6;
                background-color: #2d2d2d;
            }
        """)

        layout = QVBoxLayout(self)
        self.label = QLabel(title)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: #9ca3af; font-weight: bold;")
        layout.addWidget(self.label)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
            self.setStyleSheet("""
                QFrame {
                    border: 2px dashed #3b82f6;
                    background-color: #333333;
                }
            """)
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #4b5563;
                border-radius: 8px;
                background-color: #262626;
            }
        """)

    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #4b5563;
                border-radius: 8px;
                background-color: #262626;
            }
        """)
        urls = event.mimeData().urls()
        if urls:
            paths = [u.toLocalFile() for u in urls]
            self.filesDropped.emit(paths)


class PreprocessingWindow(QMainWindow):
    """Preprocessing window - first window user sees."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Audio Preprocessing - Faster-Whisper GUI")

        # Apply Dark Theme
        app = QApplication.instance()
        if app:
            app.setStyleSheet(DARK_THEME_QSS)

        # Apply Windows Dark Title Bar
        apply_dark_title_bar(int(self.winId()))

        self._worker: Optional[PreprocessingWorker] = None
        self._transcription_window = None  # Will hold reference to TranscriptionWindow

        # UI Components
        self.file_list: QListWidget
        self.convert_check: QCheckBox
        self.trim_check: QCheckBox
        self.normalize_check: QCheckBox
        self.noise_check: QCheckBox
        self.music_check: QCheckBox
        self.db_slider: QSlider
        self.db_label: QLabel
        self.output_edit: QLineEdit
        self.progress_bar: QProgressBar
        self.log_output: QTextEdit
        self.start_btn: QPushButton
        self.skip_btn: QPushButton
        self.cancel_btn: QPushButton

        # Settings
        self.settings = QSettings("FasterWhisperGUI", "Preprocessing")

        self._build_ui()
        self._setup_logging()
        self._load_settings()

        self.resize(DEFAULT_WIDTH, DEFAULT_HEIGHT)
        self._center_window()

    def _center_window(self) -> None:
        """Centers the window on the screen."""
        frame_gm = self.frameGeometry()
        screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
        center_point = QApplication.desktop().screenGeometry(screen).center()
        frame_gm.moveCenter(center_point)
        self.move(frame_gm.topLeft())

    def _setup_logging(self) -> None:
        """Redirect logging to the GUI text area."""
        handler = QtLogHandler(self.log_output)
        logging.getLogger().addHandler(handler)

    def _build_ui(self) -> None:
        central = QWidget(self)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Header
        header = QLabel("Audio Preprocessing")
        header.setObjectName("Header")
        header.setStyleSheet("font-size: 24px; font-weight: bold; color: #3b82f6;")
        main_layout.addWidget(header)

        # Drag & Drop Area
        self.drag_drop = DragDropWidget("Drag & Drop Audio Files Here")
        self.drag_drop.filesDropped.connect(self.on_files_dropped)
        main_layout.addWidget(self.drag_drop)

        # File List
        file_label = QLabel("Files to Preprocess:")
        file_label.setStyleSheet("font-weight: bold; color: #9ca3af;")
        main_layout.addWidget(file_label)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        main_layout.addWidget(self.file_list)

        # Buttons Row
        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add Files")
        add_btn.setObjectName("SecondaryBtn")
        add_btn.clicked.connect(self.on_add_files_clicked)
        btn_row.addWidget(add_btn)

        clear_btn = QPushButton("Clear List")
        clear_btn.setObjectName("SecondaryBtn")
        clear_btn.clicked.connect(self.file_list.clear)
        btn_row.addWidget(clear_btn)

        main_layout.addLayout(btn_row)

        # Preprocessing Options Group
        options_group = QGroupBox("Preprocessing Options")
        options_layout = QVBoxLayout(options_group)
        options_layout.setSpacing(10)

        # Convert to WAV checkbox (always enabled, cannot be unchecked)
        self.convert_check = QCheckBox("1. Convert to WAV/16kHz Mono (Always Active)")
        self.convert_check.setToolTip("Convert audio to 16kHz mono WAV format - required for Whisper AI\nThis option is always enabled and cannot be disabled.")
        self.convert_check.setChecked(True)
        self.convert_check.setEnabled(False)  # Make it non-editable
        # Style to make checkbox blue while keeping text white like other checkboxes
        self.convert_check.setStyleSheet("""
            QCheckBox:disabled {
                color: #ffffff;  /* White text color - same as other checkboxes */
            }
            QCheckBox::indicator:disabled:checked {
                background-color: #3b82f6;  /* Blue checkbox background */
                border: 2px solid #2563eb;  /* Darker blue border */
            }
        """)
        options_layout.addWidget(self.convert_check)

        # Remove noise checkbox (Step 2 - works best on full spectrum)
        self.noise_check = QCheckBox("2. Remove Background Noise")
        self.noise_check.setToolTip("Apply FFT-based noise reduction filter\nWorks best on full spectrum before frequency filtering")
        options_layout.addWidget(self.noise_check)

        # Remove music checkbox (Step 3 - before normalization)
        self.music_check = QCheckBox("3. Remove Background Music")
        self.music_check.setToolTip(
            "Isolate speech frequencies using bandpass filter (200Hz-3500Hz)\n"
            "Removes bass/music <200Hz and high frequencies >3500Hz\n"
            "Applied before normalization for accurate speech-only loudness measurement"
        )
        options_layout.addWidget(self.music_check)

        # Normalize audio checkbox with slider (Step 4 - after filtering)
        normalize_layout = QHBoxLayout()
        self.normalize_check = QCheckBox("4. Normalize Audio Volume")
        self.normalize_check.setToolTip(
            "Normalize volume to consistent level using EBU R128 standard\n"
            "Applied after filtering to measure speech-band loudness accurately\n"
            "Target: -20 LUFS (broadcast speech standard)"
        )
        self.normalize_check.stateChanged.connect(self._on_normalize_toggled)
        normalize_layout.addWidget(self.normalize_check)

        normalize_layout.addWidget(QLabel("Target:"))
        self.db_label = QLabel("-20 dB")
        self.db_label.setStyleSheet("color: #3b82f6; font-weight: bold;")
        normalize_layout.addWidget(self.db_label)

        self.db_slider = QSlider(Qt.Horizontal)
        self.db_slider.setRange(-30, 0)
        self.db_slider.setValue(-20)
        self.db_slider.setEnabled(False)
        self.db_slider.valueChanged.connect(self._on_db_changed)
        normalize_layout.addWidget(self.db_slider)
        normalize_layout.setStretch(2, 1)  # Make slider expand

        options_layout.addLayout(normalize_layout)

        # Trim silence checkbox (Step 5 - LAST, after all processing)
        self.trim_check = QCheckBox("5. Trim Silence (VAD)")
        self.trim_check.setToolTip(
            "Use Voice Activity Detection to remove ALL silence segments\n"
            "Concatenates detected speech segments, removing silence from anywhere in audio\n"
            "Applied LAST - works best on normalized, clean speech\n\n"
            "VAD Configuration:\n"
            "• min_silence_duration_ms=3000 (3 seconds)\n"
            "  - Minimum silence duration required to split audio\n"
            "  - Only silences longer than 3s will be removed\n"
            "  - More aggressive silence removal for long pauses\n\n"
            "• speech_pad_ms=1000 (1 second)\n"
            "  - Padding added before and after detected speech\n"
            "  - Adds 1s buffer on each side of detected speech\n"
            "  - Prevents cutting off beginning/end of words\n\n"
            "• threshold=0.1 (10% confidence)\n"
            "  - Probability threshold for speech detection\n"
            "  - Audio with >10% probability of speech is kept\n"
            "  - Very sensitive - may include some background noise\n\n"
            "Why last? Normalized amplitude ensures reliable threshold-based detection"
        )
        options_layout.addWidget(self.trim_check)

        main_layout.addWidget(options_group)

        # Output Folder
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Folder:"))

        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Temp Directory (Auto)")
        self.output_edit.setReadOnly(True)
        output_layout.addWidget(self.output_edit)

        output_btn = QPushButton("Browse...")
        output_btn.setObjectName("SecondaryBtn")
        output_btn.clicked.connect(self.on_select_output_clicked)
        output_layout.addWidget(output_btn)

        main_layout.addLayout(output_layout)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        # Live Logs
        log_label = QLabel("Live Logs:")
        log_label.setStyleSheet("font-weight: bold; color: #9ca3af;")
        main_layout.addWidget(log_label)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(150)
        main_layout.addWidget(self.log_output)

        # Action Buttons
        action_layout = QHBoxLayout()

        self.start_btn = QPushButton("Start Audio Cleaning")
        self.start_btn.setMinimumHeight(45)
        self.start_btn.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.start_btn.clicked.connect(self.on_start_transcription_clicked)
        action_layout.addWidget(self.start_btn)

        self.skip_btn = QPushButton("Skip Preprocessing")
        self.skip_btn.setObjectName("SecondaryBtn")
        self.skip_btn.setMinimumHeight(45)
        self.skip_btn.clicked.connect(self.on_skip_preprocessing_clicked)
        action_layout.addWidget(self.skip_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setMinimumHeight(45)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: white;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:disabled {
                background-color: #d1d5db;
                color: #9ca3af;
            }
        """)
        self.cancel_btn.clicked.connect(self.on_cancel_clicked)
        action_layout.addWidget(self.cancel_btn)

        main_layout.addLayout(action_layout)

        self.setCentralWidget(central)

    def _load_settings(self) -> None:
        """Load preprocessing settings."""
        self.convert_check.setChecked(self.settings.value("convert_wav", True, type=bool))
        self.trim_check.setChecked(self.settings.value("trim_silence", False, type=bool))
        self.normalize_check.setChecked(self.settings.value("normalize", False, type=bool))
        self.noise_check.setChecked(self.settings.value("reduce_noise", False, type=bool))
        self.music_check.setChecked(self.settings.value("remove_music", False, type=bool))

        target_db = self.settings.value("target_db", -20, type=int)
        self.db_slider.setValue(target_db)

        output_dir = self.settings.value("output_dir", "")
        if output_dir and Path(output_dir).exists():
            self.output_edit.setText(output_dir)

        self._on_normalize_toggled()

    def _save_settings(self) -> None:
        """Save preprocessing settings."""
        self.settings.setValue("convert_wav", self.convert_check.isChecked())
        self.settings.setValue("trim_silence", self.trim_check.isChecked())
        self.settings.setValue("normalize", self.normalize_check.isChecked())
        self.settings.setValue("reduce_noise", self.noise_check.isChecked())
        self.settings.setValue("remove_music", self.music_check.isChecked())
        self.settings.setValue("target_db", self.db_slider.value())
        self.settings.setValue("output_dir", self.output_edit.text())

    def _on_normalize_toggled(self) -> None:
        """Enable/disable dB slider based on normalize checkbox."""
        self.db_slider.setEnabled(self.normalize_check.isChecked())

    def _on_db_changed(self, value: int) -> None:
        """Update dB label when slider changes."""
        self.db_label.setText(f"{value} dB")

    def _add_file_item(self, path: str) -> None:
        """Helper to add a file item with numbering."""
        row = self.file_list.count()
        filename = Path(path).name
        item_text = f"{row + 1}. {filename}"

        item = QListWidgetItem(item_text)
        item.setData(Qt.UserRole, str(path))
        self.file_list.addItem(item)

    def on_files_dropped(self, paths: List[str]) -> None:
        """Handle files dropped onto the drag-drop area."""
        for p in paths:
            self._add_file_item(p)

    def on_add_files_clicked(self) -> None:
        """Handle Add Files button click."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select media files", "", MEDIA_FILTER
        )
        if paths:
            for p in paths:
                self._add_file_item(p)

    def on_select_output_clicked(self) -> None:
        """Handle output folder selection."""
        path = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", self.output_edit.text()
        )
        if path:
            self.output_edit.setText(path)

    def on_start_transcription_clicked(self) -> None:
        """Handle Start Transcription button click."""
        self._save_settings()

        # Check if any preprocessing is selected
        any_preprocessing = (
            self.convert_check.isChecked() or
            self.trim_check.isChecked() or
            self.normalize_check.isChecked() or
            self.noise_check.isChecked() or
            self.music_check.isChecked()
        )

        # Get file list
        input_files = []
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            path_str = item.data(Qt.UserRole)
            if path_str:
                path = Path(path_str)
                if path.is_file():
                    input_files.append(path)

        # If no preprocessing and no files, just open transcription window
        if not any_preprocessing or not input_files:
            self._open_transcription_window([])
            return

        # Start preprocessing
        LOGGER.info("Starting preprocessing for %d files...", len(input_files))

        # Create preprocessing config
        output_dir_str = self.output_edit.text().strip()
        output_dir = Path(output_dir_str) if output_dir_str else None

        config = PreprocessingConfig(
            convert_to_wav=self.convert_check.isChecked(),
            trim_silence=self.trim_check.isChecked(),
            normalize_audio=self.normalize_check.isChecked(),
            reduce_noise=self.noise_check.isChecked(),
            remove_music=self.music_check.isChecked(),
            target_db=float(self.db_slider.value()),
            output_dir=output_dir
        )

        # Create and start worker
        self._worker = PreprocessingWorker(input_files, config)
        self._worker.progress.connect(self.on_progress)
        self._worker.step_progress.connect(self.on_step_progress)
        self._worker.file_status.connect(self.on_file_status)
        self._worker.finished.connect(self.on_preprocessing_finished)
        self._worker.failed.connect(self.on_preprocessing_failed)

        self._set_busy(True)
        self._worker.start()

    def on_skip_preprocessing_clicked(self) -> None:
        """Handle Skip Preprocessing button click."""
        self._open_transcription_window([])

    def on_cancel_clicked(self) -> None:
        """Handle Cancel button click."""
        if self._worker:
            self._worker.request_cancel()
        LOGGER.info("Cancelling preprocessing...")

    def on_progress(self, percent: int) -> None:
        """Update progress bar."""
        self.progress_bar.setValue(percent)
        LOGGER.debug(f"Overall progress: {percent}%")

    def on_step_progress(self, filename: str, step_name: str, step_percent: int) -> None:
        """Handle step-level progress updates."""
        # Log step information for user visibility
        msg = f"[{filename}] {step_name}... ({step_percent}%)"
        LOGGER.info(msg)

    def on_file_status(self, filename: str, status: str) -> None:
        """Update file status in list."""
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            path_str = item.data(Qt.UserRole)
            if path_str and Path(path_str).name == filename:
                item.setText(f"{i+1}. {filename} [{status}]")
                self.file_list.scrollToItem(item)
                break

    def on_preprocessing_finished(self, preprocessed_files: List[Path]) -> None:
        """Handle preprocessing completion."""
        LOGGER.info("Preprocessing completed. %d files ready.", len(preprocessed_files))
        self._set_busy(False)
        self._worker = None

        # Open transcription window with preprocessed files
        self._open_transcription_window(preprocessed_files)

    def on_preprocessing_failed(self, message: str) -> None:
        """Handle preprocessing failure."""
        LOGGER.error("Preprocessing failed: %s", message)
        self._set_busy(False)
        self._worker = None

        if message != "Cancelled":
            QMessageBox.critical(self, "Error", f"Preprocessing failed:\n{message}")

    def _open_transcription_window(self, initial_files: List[Path]) -> None:
        """Open the transcription window and close this window."""
        from gui import TranscriptionWindow

        LOGGER.info("Opening transcription window with %d files...", len(initial_files))

        # Create transcription window
        self._transcription_window = TranscriptionWindow(initial_files=initial_files)
        self._transcription_window.show()

        # Close preprocessing window
        self.close()

    def _set_busy(self, busy: bool) -> None:
        """Set UI to busy/idle state."""
        self.start_btn.setEnabled(not busy)
        self.skip_btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)
        self.file_list.setEnabled(not busy)
        self.drag_drop.setVisible(not busy)
        # convert_check is always disabled, so we don't change its state
        self.trim_check.setEnabled(not busy)
        self.normalize_check.setEnabled(not busy)
        self.noise_check.setEnabled(not busy)
        self.music_check.setEnabled(not busy)
        self.db_slider.setEnabled(not busy and self.normalize_check.isChecked())
