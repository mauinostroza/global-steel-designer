"""Delegate custom — dibuja una barra de demanda/capacidad (D/C) inline en la celda."""

from PySide6.QtWidgets import QStyledItemDelegate, QStyle
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QPainter
from steeldesigner.ui.theme import OK_TEXT, ERROR_TEXT, TABLE_HEADER_BG, TEXT_PRIMARY

# Rol custom para guardar el ratio D/C numérico asociado a la celda.
RATIO_ROLE = Qt.UserRole + 1


class DemandaDelegate(QStyledItemDelegate):
    """Pinta una barra de progreso coloreada (verde/rojo) con el % de ratio D/C."""

    def paint(self, painter: QPainter, option, index):
        ratio = index.data(RATIO_ROLE)
        if ratio is None:
            super().paint(painter, option, index)
            return

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)

        if option.state & QStyle.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())

        margin = 6
        rect = QRectF(
            option.rect.x() + margin,
            option.rect.y() + margin,
            option.rect.width() - 2 * margin,
            option.rect.height() - 2 * margin,
        )

        # Fondo de la barra
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(TABLE_HEADER_BG))
        painter.drawRoundedRect(rect, 3, 3)

        # Barra proporcional (tope visual en 100%)
        color = QColor(OK_TEXT) if ratio <= 1.0 else QColor(ERROR_TEXT)
        fill_width = rect.width() * min(ratio, 1.0)
        fill_rect = QRectF(rect.x(), rect.y(), fill_width, rect.height())
        painter.setBrush(color)
        painter.drawRoundedRect(fill_rect, 3, 3)

        # Texto "NN%"
        painter.setPen(QColor(TEXT_PRIMARY))
        painter.drawText(rect, Qt.AlignCenter, f"{ratio * 100:.0f}%")

        painter.restore()

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        size.setHeight(max(size.height(), 22))
        return size
