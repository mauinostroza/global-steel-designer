"""Indicador de estado (OK/Error/Warning/Info) con colores corporativos."""

from PySide6.QtWidgets import QLabel, QHBoxLayout, QFrame, QSizePolicy
from PySide6.QtCore import Qt
from steeldesigner.ui.theme import (
    OK_BG, OK_TEXT, ERROR_BG, ERROR_TEXT,
    WARN_BG, WARN_TEXT, INFO_BG, INFO_TEXT,
    RADIUS_SM, FONT_SIZE_SM,
)


class StatusIndicator(QFrame):
    """Indicador visual de estado tipo badge."""

    STATUS_STYLES = {
        "ok": (OK_BG, OK_TEXT, "✓"),
        "error": (ERROR_BG, ERROR_TEXT, "✗"),
        "warn": (WARN_BG, WARN_TEXT, "▲"),
        "info": (INFO_BG, INFO_TEXT, "ℹ"),
    }

    def __init__(self, status: str = "info", text: str = "", parent=None):
        super().__init__(parent)
        self._status = status
        self._text = text

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(8, 3, 8, 3)
        self._layout.setSpacing(4)

        self._icon_label = QLabel()
        self._text_label = QLabel(text)

        self._icon_label.setStyleSheet("border: none; font-weight: 600;")
        self._text_label.setStyleSheet("border: none;")

        self._layout.addWidget(self._icon_label)
        self._layout.addWidget(self._text_label)

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.apply_style()

    def apply_style(self):
        bg, fg, icon = self.STATUS_STYLES.get(self._status, self.STATUS_STYLES["info"])
        self.setStyleSheet(f"""
            StatusIndicator {{
                background: {bg};
                border-radius: {RADIUS_SM};
                padding: 0;
            }}
            QLabel {{
                color: {fg};
                font-size: {FONT_SIZE_SM};
                font-weight: 500;
            }}
        """)
        self._icon_label.setText(icon)

    def set_status(self, status: str, text: str = ""):
        self._status = status
        if text:
            self._text = text
            self._text_label.setText(text)
        self.apply_style()
