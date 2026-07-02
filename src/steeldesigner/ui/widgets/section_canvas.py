"""
Canvas genérico para dibujar secciones de acero usando las primitivas
del módulo catalog.drawer (sin dependencias PyQt).
"""
from __future__ import annotations
from typing import Optional
from PySide6.QtWidgets import QWidget, QSizePolicy
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont

from steeldesigner.ui.theme import BRAND, BORDER, TEXT_PRIMARY, TEXT_SECONDARY, BG_CARD

try:
    from steeldesigner.catalog.drawer import draw_section, bounding_box
    _DRAWER_OK = True
except Exception:
    _DRAWER_OK = False

_LAYER_COLORS = {
    "outline":    "#1D1D1F",
    "fill":       "#0066CC",
    "dimension":  "#6E6E73",
    "centerline": "#D0021B",
    "text":       "#1D1D1F",
}


class SectionCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._section = None
        self._primitives = []
        self._bbox = (0, 0, 100, 100)

    def set_section(self, section) -> None:
        self._section = section
        self._primitives = []
        if not _DRAWER_OK or section is None:
            self.update()
            return
        try:
            prims = draw_section(section)
            self._primitives = prims
            self._bbox = bounding_box(prims)
        except Exception:
            self._primitives = []
        self.update()

    def clear(self) -> None:
        self._section = None
        self._primitives = []
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(BG_CARD))
        if not self._section or not self._primitives:
            painter.setPen(QColor(TEXT_SECONDARY))
            painter.drawText(self.rect(), Qt.AlignCenter, "Selecciona un perfil del catálogo")
            return
        self._render(painter)

    def _render(self, painter: QPainter) -> None:
        xmin, ymin, xmax, ymax = self._bbox
        sec_w = xmax - xmin or 1.0
        sec_h = ymax - ymin or 1.0
        w, h, margin = self.width(), self.height(), 20
        scale = min((w - 2 * margin) / sec_w, (h - 2 * margin) / sec_h)
        off_x = (w - sec_w * scale) / 2 - xmin * scale
        off_y_flip = h - ((h - sec_h * scale) / 2 - ymin * scale)

        def tx(x): return x * scale + off_x
        def ty(y): return off_y_flip - y * scale

        for prim in self._primitives:
            ptype = type(prim).__name__
            layer = getattr(prim, "layer", "outline")
            color = QColor(_LAYER_COLORS.get(layer, "#000000"))

            if ptype == "Line":
                pen = QPen(color, 1.5 if layer == "outline" else 0.8)
                if layer == "centerline":
                    pen.setStyle(Qt.DashLine)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawLine(QPointF(tx(prim.x1), ty(prim.y1)),
                                 QPointF(tx(prim.x2), ty(prim.y2)))

            elif ptype == "Circle":
                pen = QPen(color, 1.5)
                painter.setPen(pen)
                fill = QColor(BRAND); fill.setAlpha(25)
                painter.setBrush(QBrush(fill))
                cx, cy, r = tx(prim.cx), ty(prim.cy), prim.r * scale
                painter.drawEllipse(QPointF(cx, cy), r, r)
                painter.setBrush(Qt.NoBrush)

            elif ptype == "Arc":
                pen = QPen(color, 1.5)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                cx, cy, r = tx(prim.cx), ty(prim.cy), prim.r * scale
                rect = QRectF(cx - r, cy - r, 2 * r, 2 * r)
                painter.drawArc(rect, int(prim.start_deg * 16), int(prim.span_deg * 16))

            elif ptype == "Dimension":
                pen = QPen(color, 0.7)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawLine(QPointF(tx(prim.x1), ty(prim.y1)),
                                 QPointF(tx(prim.x2), ty(prim.y2)))
                painter.setFont(QFont("Segoe UI", 7))
                painter.setPen(QColor(_LAYER_COLORS["text"]))
                mx = (tx(prim.x1) + tx(prim.x2)) / 2
                my = (ty(prim.y1) + ty(prim.y2)) / 2
                painter.drawText(QPointF(mx + 2, my - 2), prim.label)
