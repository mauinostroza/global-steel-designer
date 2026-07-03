"""Página de Resultados — historial de diseños, comparaciones y exportación."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QFileDialog, QAbstractItemView, QTabWidget, QCheckBox,
)
from PySide6.QtCore import Qt
from steeldesigner.ui.theme import (
    BG_CONTENT, TEXT_PRIMARY, TEXT_SECONDARY, BRAND, OK_BG, OK_TEXT,
    ERROR_BG, ERROR_TEXT, BORDER,
)
from steeldesigner.ui.widgets.card import Card, MetricCard
from steeldesigner.ui.widgets.status_indicator import StatusIndicator
from steeldesigner.ui.widgets.demanda_delegate import DemandaDelegate, RATIO_ROLE


class ResultsPage(QWidget):
    """Historial de resultados de diseño con comparación y exportación."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results: list[dict] = []
        self.setStyleSheet(f"background: {BG_CONTENT};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Header
        header = QLabel("Historial de Resultados")
        header.setStyleSheet(f"font-size: 18px; font-weight: 500; color: {TEXT_PRIMARY}; border: none;")
        layout.addWidget(header)

        # Botones de acción
        btn_row = QHBoxLayout()
        self._btn_clear = QPushButton("Limpiar historial")
        self._btn_clear.setObjectName("btnSecondary")
        self._btn_clear.clicked.connect(self._clear_history)
        btn_row.addWidget(self._btn_clear)

        self._btn_export = QPushButton("  Exportar todo")
        self._btn_export.setObjectName("btnPrimary")
        self._btn_export.clicked.connect(self._export_all)
        btn_row.addWidget(self._btn_export)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Tabla de resultados
        self._results_table = QTableWidget()
        self._results_table.setColumnCount(12)
        self._results_table.setHorizontalHeaderLabels([
            "#", "Perfil", "L (mm)", "Pu (kN)", "φPn (kN)",
            "Ratio D/C", "(KL/r)eff", "(KL/r)design", "Fcr (MPa)", "Modo", "Conexión", "Estado",
        ])
        self._results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._results_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._results_table.setAlternatingRowColors(True)
        self._results_table.horizontalHeader().setStretchLastSection(True)
        self._results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._results_table.verticalHeader().setVisible(False)
        self._results_table.itemClicked.connect(self._on_result_clicked)
        self._results_table.setItemDelegateForColumn(5, DemandaDelegate(self._results_table))
        self._results_table.setStyleSheet("""
            QTableWidget {
                border: 0.5px solid #E0DDD8;
                border-radius: 8px;
                background: #FFFFFF;
                alternate-background-color: #FAFAF8;
                gridline-color: #F0EFED;
                selection-background-color: #FFF5F5;
                selection-color: #1C1C1F;
            }
            QTableWidget::item { padding: 4px 10px; font-size: 13px; border-bottom: 0.5px solid #F0EFED; }
            QHeaderView::section {
                background: #F0EFED; color: #444444;
                font-size: 11px; font-weight: 500; letter-spacing: 0.5px;
                text-transform: uppercase; padding: 8px 12px;
                border: none; border-bottom: 1px solid #E0DDD8;
            }
        """)
        layout.addWidget(self._results_table, 1)

        # Detalle del resultado seleccionado
        self._detail_card = Card("Detalle del resultado seleccionado")
        detail_header = QHBoxLayout()
        self._detail_status = StatusIndicator("info", "Selecciona un resultado de la tabla")
        detail_header.addWidget(self._detail_status)
        detail_header.addStretch()
        self._detail_card.add_layout(detail_header)

        # Métricas en el detalle
        metrics_layout = QHBoxLayout()
        self._detail_phiPn = MetricCard("φPn", "—", "kN")
        self._detail_ratio = MetricCard("Ratio", "—")
        self._detail_KLeff = MetricCard("(KL/r)eff", "—")
        self._detail_KLdesign = MetricCard("(KL/r)design", "—")
        self._detail_Pu = MetricCard("Pu", "—", "kN")
        self._detail_mode = MetricCard("Modo", "—")
        for m in [self._detail_phiPn, self._detail_ratio, self._detail_KLeff, self._detail_KLdesign, self._detail_Pu, self._detail_mode]:
            metrics_layout.addWidget(m)
        self._detail_card.add_layout(metrics_layout)

        layout.addWidget(self._detail_card)

        # Info label
        self._info_label = QLabel("Los resultados de diseño aparecerán aquí después de cada cálculo.")
        self._info_label.setStyleSheet(f"color: {TEXT_SECONDARY}; border: none;")
        layout.addWidget(self._info_label)

    def add_result(self, result: dict):
        """Agrega un resultado al historial y a la tabla."""
        self._results.append(result)
        self._update_table()

    def _update_table(self):
        """Refresca la tabla con todos los resultados."""
        self._results_table.setRowCount(len(self._results))
        for i, r in enumerate(self._results):
            self._results_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self._results_table.setItem(i, 1, QTableWidgetItem(r.get("section", "—")))
            self._results_table.setItem(i, 2, QTableWidgetItem(f"{r.get('L/mm', 0) if 'L/mm' in r else '—':.0f}" if isinstance(r.get('L/mm', '—'), (int, float)) else str(r.get('L/mm', '—'))))
            self._results_table.setItem(i, 3, QTableWidgetItem(f"{r.get('Pu_kN', 0):.1f}"))
            self._results_table.setItem(i, 4, QTableWidgetItem(f"{r.get('phiPn_kN', 0):.2f}"))
            ratio_item = QTableWidgetItem("")
            ratio_item.setData(RATIO_ROLE, r.get("ratio", 0))
            self._results_table.setItem(i, 5, ratio_item)
            self._results_table.setItem(i, 6, QTableWidgetItem(f"{r.get('(KL/r)eff', 0):.1f}"))
            self._results_table.setItem(i, 7, QTableWidgetItem(f"{r.get('(KL/r)design', 0):.1f}"))
            self._results_table.setItem(i, 8, QTableWidgetItem(f"{r.get('Fcr_MPa', 0):.1f}"))
            self._results_table.setItem(i, 9, QTableWidgetItem(r.get("mode", "—")))
            self._results_table.setItem(i, 10, QTableWidgetItem(r.get("conn_leg", "—")))
            ok = r.get("ok", False)
            self._results_table.setItem(i, 11, QTableWidgetItem("✓ OK" if ok else "✗ NO OK"))

        self._results_table.resizeColumnsToContents()
        self._info_label.setText(f"{len(self._results)} resultado(s) en el historial.")

    def _on_result_clicked(self, item):
        """Muestra detalle del resultado seleccionado."""
        row = item.row()
        if row >= len(self._results):
            return
        r = self._results[row]

        ok = r.get("ok", False)
        self._detail_status.set_status(
            "ok" if ok else "error",
            f"✓ OK — Reserva {r.get('reserve_pct', 0):.1f}%" if ok else f"✗ NO OK — Déficit {r.get('deficit_pct', 0):.1f}%"
        )
        self._detail_phiPn.set_value(f"{r.get('phiPn_kN', 0):.2f}")
        self._detail_ratio.set_value(f"{r.get('ratio', 0):.4f}")
        self._detail_KLeff.set_value(f"{r.get('(KL/r)eff', 0):.1f}")
        self._detail_KLdesign.set_value(f"{r.get('(KL/r)design', 0):.1f}")
        self._detail_Pu.set_value(f"{r.get('Pu_kN', 0):.2f}")
        self._detail_mode.set_value(r.get("mode", "—"))

    def _clear_history(self):
        """Limpia todo el historial."""
        if not self._results:
            return
        reply = QMessageBox.question(self, "Limpiar", "¿Eliminar todo el historial?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self._results.clear()
            self._results_table.setRowCount(0)
            self._info_label.setText("Historial limpiado.")
            self._detail_status.set_status("info", "Selecciona un resultado de la tabla")

    def _export_all(self):
        """Exporta todos los resultados a CSV."""
        if not self._results:
            QMessageBox.information(self, "Exportar", "No hay resultados para exportar.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Guardar CSV", "resultados_diseno.csv", "CSV (*.csv)")
        if not path:
            return

        try:
            import csv
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "#", "Perfil", "Pu (kN)", "φPn (kN)", "Ratio",
                    "(KL/r)eff", "(KL/r)design", "Fcr (MPa)", "Modo", "OK",
                    "A (mm²)", "rx (mm)", "ry (mm)", "rz (mm)",
                    "Q", "b₁/t", "b₂/t", "Fórmula",
                ])
                for i, r in enumerate(self._results):
                    writer.writerow([
                        i + 1,
                        r.get("section", ""),
                        r.get("Pu_kN", 0),
                        r.get("phiPn_kN", 0),
                        r.get("ratio", 0),
                        r.get("(KL/r)eff", 0),
                        r.get("(KL/r)design", 0),
                        r.get("Fcr_MPa", 0),
                        r.get("mode", ""),
                        "OK" if r.get("ok") else "NO OK",
                        r.get("A_mm2", ""),
                        r.get("rx_mm", ""),
                        r.get("ry_mm", ""),
                        r.get("rz_mm", ""),
                        r.get("Q", ""),
                        r.get("b1/t", ""),
                        r.get("b2/t", ""),
                        r.get("formula", ""),
                    ])
            QMessageBox.information(self, "Exportar", f"{len(self._results)} resultado(s) exportados a:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error exportando: {e}")

    def export_data(self):
        """Método llamado desde el botón Exportar del Topbar."""
        self._export_all()
