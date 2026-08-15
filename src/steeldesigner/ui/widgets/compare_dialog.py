"""
Diálogo de comparación de 2-4 perfiles.

Usa catalog.comparator.Comparison (tabla agrupada con resaltado de
diferencias) y agrega un gráfico de barras (PySide6.QtCharts, sin
dependencias nuevas) comparando Peso/A/Ix/Zx entre los perfiles elegidos.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QLabel, QFileDialog, QMessageBox, QApplication,
    QTabWidget, QWidget,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor, QPainter
from PySide6.QtCharts import (
    QChart, QChartView, QBarSeries, QBarSet, QBarCategoryAxis, QValueAxis,
)

from steeldesigner.catalog.comparator import Comparison
from steeldesigner.core.section_geometry import apply_to_section
from steeldesigner.ui.theme import BRAND, TEXT_PRIMARY, TEXT_SECONDARY, BORDER, BG_CARD

_HIGH_COLOR = QColor("#FFF8E1")     # diferencia >5% vs promedio
_EXTREME_COLOR = QColor("#FFE0B2")  # diferencia >20% vs promedio

# Propiedades graficadas en el bar chart (label, atributo Section, unidad)
_CHART_PROPS = [
    ("Peso (kg/m)", "weight_kg_m", 1.0),
    ("A (cm²)", "area_mm2", 100.0),
    ("Ix (cm⁴)", "Ix_mm4", 1e4),
    ("Zx (cm³)", "Zx_mm3", 1e3),
]


class CompareDialog(QDialog):
    def __init__(self, sections: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Comparación de perfiles")
        self.resize(900, 640)
        for sec in sections:
            apply_to_section(sec)
        self._comparison = Comparison.create(sections)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)

        title = QLabel(
            "Comparando: " + " · ".join(
                s.designation_modern or s.designation_legacy or "?"
                for s in self._comparison.sections
            )
        )
        title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        title.setStyleSheet(f"color:{TEXT_PRIMARY};")
        root.addWidget(title)

        tabs = QTabWidget()
        tabs.addTab(self._build_table_tab(), "Tabla")
        tabs.addTab(self._build_chart_tab(), "Gráfico")
        root.addWidget(tabs, stretch=1)

        btn_row = QHBoxLayout()
        btn_excel = QPushButton("Exportar a Excel…")
        btn_excel.clicked.connect(self._export_excel)
        btn_tsv = QPushButton("Copiar TSV")
        btn_tsv.clicked.connect(self._copy_tsv)
        btn_close = QPushButton("Cerrar")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_excel)
        btn_row.addWidget(btn_tsv)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        root.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Tabla
    # ------------------------------------------------------------------
    def _build_table_tab(self) -> QWidget:
        c = self._comparison
        w = QWidget()
        lv = QVBoxLayout(w)
        lv.setContentsMargins(0, 6, 0, 0)

        table = QTableWidget(c.n_rows, 1 + c.n_sections)
        headers = ["Propiedad"] + [
            s.designation_modern or s.designation_legacy or "?" for s in c.sections
        ]
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for i in range(1, 1 + c.n_sections):
            table.horizontalHeader().setSectionResizeMode(i, QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(table.NoEditTriggers)
        table.setStyleSheet(f"QTableWidget {{ font-size: 12px; gridline-color: {BORDER}; }}")

        for row_idx, ((cat, label, _key), row_cells) in enumerate(zip(c.rows, c.cells)):
            label_item = QTableWidgetItem(f"{label}")
            label_item.setToolTip(cat)
            table.setItem(row_idx, 0, label_item)
            for col_idx, cell in enumerate(row_cells):
                item = QTableWidgetItem(cell.formatted)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if cell.highlight == "extreme":
                    item.setBackground(_EXTREME_COLOR)
                elif cell.highlight == "high":
                    item.setBackground(_HIGH_COLOR)
                table.setItem(row_idx, 1 + col_idx, item)

        lv.addWidget(table)
        legend = QLabel(
            "🟧 Difiere >20% del promedio de la fila &nbsp;&nbsp; 🟨 Difiere >5%"
        )
        legend.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:11px;")
        lv.addWidget(legend)
        return w

    # ------------------------------------------------------------------
    # Gráfico
    # ------------------------------------------------------------------
    def _build_chart_tab(self) -> QWidget:
        w = QWidget()
        lv = QVBoxLayout(w)
        lv.setContentsMargins(0, 6, 0, 0)

        chart = QChart()
        chart.setTitle("Comparación de propiedades")
        chart.setAnimationOptions(QChart.SeriesAnimations)

        names = [
            s.designation_modern or s.designation_legacy or "?"
            for s in self._comparison.sections
        ]

        series = QBarSeries()
        for label, attr, divisor in _CHART_PROPS:
            bar_set = QBarSet(label)
            for sec in self._comparison.sections:
                v = getattr(sec, attr, None) or 0.0
                bar_set.append(v / divisor)
            series.append(bar_set)
        chart.addSeries(series)

        axis_x = QBarCategoryAxis()
        axis_x.append(names)
        chart.addAxis(axis_x, Qt.AlignBottom)
        series.attachAxis(axis_x)

        axis_y = QValueAxis()
        axis_y.setTitleText("Valor (unidades normalizadas, ver leyenda)")
        chart.addAxis(axis_y, Qt.AlignLeft)
        series.attachAxis(axis_y)

        chart.legend().setVisible(True)
        chart.legend().setAlignment(Qt.AlignBottom)

        view = QChartView(chart)
        view.setRenderHint(QPainter.Antialiasing)
        lv.addWidget(view)
        return w

    # ------------------------------------------------------------------
    # Exportación
    # ------------------------------------------------------------------
    def _export_excel(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar comparación", "comparacion_perfiles.xlsx", "Excel (*.xlsx)"
        )
        if not path:
            return
        try:
            self._comparison.to_excel(path)
        except ImportError as exc:
            QMessageBox.warning(self, "Falta dependencia", str(exc))
        except Exception as exc:
            QMessageBox.critical(self, "Error al exportar", str(exc))

    def _copy_tsv(self):
        QApplication.clipboard().setText(self._comparison.to_tsv())
