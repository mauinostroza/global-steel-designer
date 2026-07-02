"""Sección colapsable — condensa formularios secundarios sin perder la información."""

from PySide6.QtWidgets import QFrame, QVBoxLayout, QToolButton, QWidget, QSizePolicy
from PySide6.QtCore import Qt
from steeldesigner.ui.theme import TEXT_PRIMARY, TEXT_SECONDARY, BORDER, FONT_SIZE_SM


class CollapsibleSection(QFrame):
    """Cabecera clickeable (QToolButton) que expande/colapsa un widget de contenido."""

    def __init__(self, titulo: str, contenido: QWidget, collapsed: bool = True, parent=None):
        super().__init__(parent)
        self.setObjectName("collapsibleSection")
        self.setStyleSheet("""
            #collapsibleSection {
                border: none;
            }
        """)

        self._contenido = contenido
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._toggle = QToolButton()
        self._toggle.setText(titulo)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(not collapsed)
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.DownArrow if not collapsed else Qt.RightArrow)
        self._toggle.setCursor(Qt.PointingHandCursor)
        self._toggle.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._toggle.setStyleSheet(f"""
            QToolButton {{
                border: none;
                background: transparent;
                color: {TEXT_SECONDARY};
                font-size: {FONT_SIZE_SM};
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                padding: 4px 0;
            }}
            QToolButton:hover {{
                color: {TEXT_PRIMARY};
            }}
        """)
        self._toggle.clicked.connect(self._on_toggled)

        layout.addWidget(self._toggle)
        layout.addWidget(contenido)
        contenido.setVisible(not collapsed)

    def _on_toggled(self):
        expanded = self._toggle.isChecked()
        self._toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._contenido.setVisible(expanded)

    def set_expanded(self, expanded: bool):
        self._toggle.setChecked(expanded)
        self._on_toggled()
