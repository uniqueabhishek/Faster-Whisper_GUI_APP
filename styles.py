"""
Modern Dark Theme for Faster-Whisper GUI.
Colors:
- Background: #1e1e1e (Dark Grey)
- Surface: #2d2d2d (Lighter Grey for cards/inputs)
- Primary: #3b82f6 (Bright Blue)
- Text: #e5e7eb (Off-white)
- Border: #404040
"""

import ctypes

def apply_dark_title_bar(window_handle):
    """
    Apply Windows Immersive Dark Mode to the title bar.
    Requires Windows 10 (Build 1903+) or Windows 11.
    """
    try:
        # DWMWA_USE_IMMERSIVE_DARK_MODE = 20 (Windows 10 2004+ / Windows 11)
        DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        set_window_attribute = ctypes.windll.dwmapi.DwmSetWindowAttribute
        hwnd = window_handle

        # If it's a PyQt window, we might need the underlying HWND
        # But usually passing int(winId()) works.

        rendering_policy = DWMWA_USE_IMMERSIVE_DARK_MODE
        value = ctypes.c_int(2) # 2 = True (technically boolean, but 1 or non-zero)
        # Actually for this attribute, TRUE (1) is needed.
        value = ctypes.c_int(1)

        set_window_attribute(
            hwnd,
            rendering_policy,
            ctypes.byref(value),
            ctypes.sizeof(value)
        )
    except Exception:
        # Fail silently if not on Windows or older version
        pass

DARK_THEME_QSS = """
/* Main Window */
QMainWindow, QWidget {
    background-color: #1e1e1e;
    color: #e5e7eb;
    font-family: 'Segoe UI', 'Roboto', sans-serif;
    font-size: 14px;
}

/* Group Boxes / Cards */
QGroupBox {
    background-color: #252526;
    border: 1px solid #3f3f46;
    border-radius: 8px;
    margin-top: 1em;
    padding-top: 10px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: #9ca3af;
}

/* Inputs */
QLineEdit {
    background-color: #2d2d2d;
    border: 1px solid #404040;
    border-radius: 6px;
    padding: 8px;
    color: #ffffff;
    selection-background-color: #3b82f6;
}
QLineEdit:focus {
    border: 1px solid #3b82f6;
    background-color: #333333;
}
QLineEdit:read-only {
    background-color: #262626;
    color: #9ca3af;
}

/* ComboBox */
QComboBox {
    background-color: #2d2d2d;
    border: 1px solid #404040;
    border-radius: 6px;
    padding: 8px;
    color: #ffffff;
}
QComboBox::drop-down {
    border: none;
}
QComboBox::down-arrow {
    image: url(assets/chevron_down.svg);
    width: 12px;
    height: 12px;
    margin-right: 10px;
}
QComboBox QAbstractItemView {
    background-color: #2d2d2d;
    color: #ffffff;
    selection-background-color: #3b82f6;
    border: 1px solid #404040;
}

/* Buttons */
QPushButton {
    background-color: #3b82f6;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #2563eb;
}
QPushButton:pressed {
    background-color: #1d4ed8;
}
QPushButton:disabled {
    background-color: #4b5563;
    color: #9ca3af;
}

/* Secondary Buttons (Browse, etc) */
QPushButton#SecondaryBtn {
    background-color: #4b5563;
}
QPushButton#SecondaryBtn:hover {
    background-color: #6b7280;
}

/* Text Area (Logs/Output) */
QTextEdit {
    background-color: #111827; /* Very dark for code/logs */
    border: 1px solid #374151;
    border-radius: 6px;
    color: #10b981; /* Matrix green-ish for logs, or just white */
    font-family: 'Consolas', 'Monospace';
    font-size: 13px;
    padding: 8px;
}

/* Progress Bar */
QProgressBar {
    border: none;
    background-color: #374151;
    border-radius: 4px;
    text-align: center;
    height: 8px;
}
QProgressBar::chunk {
    background-color: #3b82f6;
    border-radius: 4px;
}

/* Scrollbars */
QScrollBar:vertical {
    border: none;
    background: #1e1e1e;
    width: 10px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #4b5563;
    min-height: 20px;
    border-radius: 5px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* List Widget */
QListWidget {
    background-color: #2d2d2d;
    border: 1px solid #404040;
    border-radius: 6px;
    padding: 5px;
}
QListWidget::item {
    padding: 5px;
    border-bottom: 1px solid #3f3f46;
}
QListWidget::item:selected {
    background-color: #374151;
    border-radius: 4px;
}

/* Labels */
QLabel {
    color: #d1d5db;
}
QLabel#Header {
    font-size: 18px;
    font-weight: bold;
    color: #ffffff;
    margin-bottom: 10px;
}
"""
