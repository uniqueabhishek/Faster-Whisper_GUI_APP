"""Application entry point for Faster-Whisper GUI."""

from __future__ import annotations

import sys
import logging
import traceback

# Fix for DLL load failed error: onnxruntime must be imported before PyQt5
try:
    import onnxruntime
except ImportError:
    pass

from PyQt5.QtWidgets import QApplication, QMessageBox

from preprocessing_gui import PreprocessingWindow
from workers import EXECUTOR


def exception_hook(exctype, value, tb):
    """Global exception handler to catch unhandled exceptions."""
    error_msg = ''.join(traceback.format_exception(exctype, value, tb))
    logging.error("Unhandled exception:\n%s", error_msg)
    print(f"\n\nUNHANDLED EXCEPTION:\n{error_msg}")

    # Try to show error dialog if possible
    try:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Fatal Error")
        msg.setText(f"An unhandled error occurred:\n\n{str(value)}")
        msg.setDetailedText(error_msg)
        msg.exec_()
    except:
        pass


def main() -> None:
    # Install global exception hook
    sys.excepthook = exception_hook

    # Setup logging to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("whisper_gui_debug.log", mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Application starting...")

    try:
        app = QApplication(sys.argv)

        # Show preprocessing window first
        window = PreprocessingWindow()
        window.show()

        sys.exit(app.exec())
    except Exception:
        logging.exception("Fatal error in application")
        print(f"\n\nFATAL ERROR:\n{traceback.format_exc()}")
        input("Press Enter to exit...")
    finally:
        # Ensure all worker threads terminate cleanly
        EXECUTOR.shutdown(wait=False)


if __name__ == "__main__":
    main()
