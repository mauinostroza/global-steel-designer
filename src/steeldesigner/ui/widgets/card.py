"""Widget de tarjeta con estilo corporativo (fondo blanco, borde #E0DDD8, radio 8px)."""

from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel
from PySide6.QtCore import Qt
from steeldesigner.ui.theme import BG_CARD, BORDER, TEXT_PRIMARY, RADIUS_MD, FONT_SIZE_SM


class Card(QFrame):
    """Tarjeta blanca con borde sutil para contener widgets."""

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setStyleSheet(f"""
            Card {{
                background: {BG_CARD};
                border: 0.5px solid {BORDER};
                border-radius: {RADIUS_MD};
            }}
        """)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(16, 12, 16, 12)
        self._layout.setSpacing(8)

        if title:
            self._title_label = QLabel(title)
            self._title_label.setStyleSheet(f"""
                QLabel {{
                    color: #444444;
                    font-size: {FONT_SIZE_SM};
                    font-weight: 500;
                    padding: 0;
                    border: none;
                }}
            """)
            self._layout.addWidget(self._title_label)

    def add_widget(self, widget):
        """Agrega un widget al contenido de la tarjeta."""
        self._layout.addWidget(widget)

    def add_layout(self, layout):
        """Agrega un layout al contenido de la tarjeta."""
        self._layout.addLayout(layout)

    def clear(self):
        """Elimina todos los widgets del contenido."""
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()


class MetricCard(Card):
    """Tarjeta para mostrar una métrica (título + valor grande)."""

    def __init__(self, title: str, value: str, unit: str = "", parent=None):
        super().__init__(title="", parent=parent)
        self._title = QLabel(title)
        self._title.setStyleSheet(f"""
            QLabel {{
                color: #444444;
                font-size: {FONT_SIZE_SM};
                font-weight: 500;
                border: none;
            }}
        """)
        self._value = QLabel(value)
        self._value.setStyleSheet(f"""
            QLabel {{
                color: {TEXT_PRIMARY};
                font-size: 22px;
                font-weight: 500;
                border: none;
            }}
        """)
        self._unit = QLabel(unit)
        self._unit.setStyleSheet(f"""
            QLabel {{
                color: #666662;
                font-size: {FONT_SIZE_SM};
                border: none;
            }}
        """)

        self._layout.addWidget(self._title)
        self._layout.addWidget(self._value)
        if unit:
            self._layout.addWidget(self._unit)

    def set_value(self, value: str):
        self._value.setText(value)

    def set_color(self, color: str):
        self._value.setStyleSheet(f"color:{color}; font-size:22px; font-weight:500; border:none;")
