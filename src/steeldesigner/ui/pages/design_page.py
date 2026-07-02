"""
Página de diseño — workspace 3 columnas: inputs | dibujo sección | resultados.

Recibe una Section del catálogo (vía load_section) o permite ingresar
manualmente material, longitudes y demandas. Llama a EngineFacade.run()
y muestra MetricCards + tabla de verificaciones + informe HTML.
"""
from __future__ import annotations

import csv
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QLabel,
    QComboBox, QDoubleSpinBox, QFormLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QFileDialog, QAbstractItemView, QTabWidget,
    QListWidget, QCheckBox, QTextBrowser, QSizePolicy, QScrollArea,
    QSplitter,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from steeldesigner.ui.theme import (
    BRAND, BG_SURFACE, BG_CARD, TEXT_PRIMARY, TEXT_SECONDARY, BORDER,
    RADIUS_MD, FONT_SIZE_BASE, FONT_SIZE_SM,
)
from steeldesigner.ui.widgets.section_canvas import SectionCanvas
from steeldesigner.ui.widgets.card import MetricCard
from steeldesigner.ui.widgets.status_indicator import StatusIndicator

from steeldesigner.core.engine_facade import EngineFacade, DesignInputs
from steeldesigner.core.section_adapter import section_props_dict, family_type


# ── style constants ──────────────────────────────────────────────────────────
_GRP = f"""
QGroupBox {{
    background: {BG_CARD};
    border: 0.5px solid {BORDER};
    border-radius: {RADIUS_MD};
    margin-top: 12px;
    padding: 14px 12px 10px 12px;
    color: {TEXT_PRIMARY};
    font-weight: normal;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 8px;
    color: {TEXT_PRIMARY};
    font-weight: 500;
}}
"""

_BTN = f"""
QPushButton {{
    background: {BRAND};
    color: white;
    border: none;
    border-radius: {RADIUS_MD};
    padding: 7px 14px;
    font-size: {FONT_SIZE_BASE};
    font-weight: 600;
}}
QPushButton:hover {{ background: #0051A8; }}
QPushButton:disabled {{ background: {BORDER}; color: {TEXT_SECONDARY}; }}
"""

_BTN2 = f"""
QPushButton {{
    background: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: {RADIUS_MD};
    padding: 7px 14px;
    font-size: {FONT_SIZE_BASE};
}}
QPushButton:hover {{ background: #F0EFED; }}
"""

_TABLE = f"""
QTableWidget {{
    border: 0.5px solid {BORDER};
    border-radius: 8px;
    background: #FFFFFF;
    alternate-background-color: #FAFAF8;
    gridline-color: #F0EFED;
}}
QTableWidget::item {{ padding: 4px 8px; font-size: 11px; }}
QHeaderView::section {{
    background: #F0EFED; color: #444;
    font-size: 10px; font-weight: 500;
    padding: 6px 8px; border: none;
    border-bottom: 1px solid {BORDER};
}}
"""

_FACADE = EngineFacade()


class DesignPage(QWidget):
    """Pantalla de diseño AISC 360-22."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._section = None
        self._last_result = None
        self._build_ui()

    # ── public API ───────────────────────────────────────────────────────────

    def load_section(self, section) -> None:
        """Carga un perfil recibido desde CataloguePage."""
        self._section = section
        self._show_section_info(section)
        self._canvas.set_section(section)
        self._status.set_status("info", f"Perfil cargado: {section.designation_modern or section.designation_legacy}")
        self._btn_calc.setEnabled(True)

    # ── build UI ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_center())
        splitter.addWidget(self._build_right())
        splitter.setSizes([320, 280, 500])

    # ── LEFT panel ───────────────────────────────────────────────────────────

    def _build_left(self) -> QWidget:
        w = QWidget()
        w.setMinimumWidth(290)
        w.setMaximumWidth(360)
        lv = QVBoxLayout(w)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(8)

        tabs = QTabWidget()
        tabs.addTab(self._build_tab_section(), "Perfil")
        tabs.addTab(self._build_tab_loads(), "Cargas")
        tabs.addTab(self._build_tab_sap(), "SAP2000")
        lv.addWidget(tabs)
        return w

    def _build_tab_section(self) -> QWidget:
        tab = QWidget()
        lv = QVBoxLayout(tab)
        lv.setContentsMargins(8, 8, 8, 8)
        lv.setSpacing(8)

        # section info card
        grp = QGroupBox("Perfil activo")
        grp.setStyleSheet(_GRP)
        gv = QVBoxLayout(grp)
        self._sec_info = QLabel("— Sin perfil seleccionado —")
        self._sec_info.setWordWrap(True)
        self._sec_info.setStyleSheet(f"font-size:11px; color:{TEXT_PRIMARY};")
        gv.addWidget(self._sec_info)
        lv.addWidget(grp)

        # material
        mat_grp = QGroupBox("Material")
        mat_grp.setStyleSheet(_GRP)
        mf = QFormLayout(mat_grp)
        mf.setSpacing(6)

        self._spin_Fy = self._spin(100, 1000, 250, " MPa")
        self._spin_Fu = self._spin(200, 1500, 400, " MPa")
        mf.addRow("Fy:", self._spin_Fy)
        mf.addRow("Fu:", self._spin_Fu)
        lv.addWidget(mat_grp)

        # method
        meth_grp = QGroupBox("Método")
        meth_grp.setStyleSheet(_GRP)
        mv = QVBoxLayout(meth_grp)
        self._combo_method = QComboBox()
        self._combo_method.addItems(["LRFD (φ)", "ASD (Ω)"])
        mv.addWidget(self._combo_method)
        lv.addWidget(meth_grp)

        lv.addStretch()
        return tab

    def _build_tab_loads(self) -> QWidget:
        tab = QWidget()
        lv = QVBoxLayout(tab)
        lv.setContentsMargins(8, 8, 8, 8)
        lv.setSpacing(8)

        geom_grp = QGroupBox("Longitudes efectivas")
        geom_grp.setStyleSheet(_GRP)
        gf = QFormLayout(geom_grp)
        gf.setSpacing(6)

        self._spin_Lx = self._spin(0, 100000, 3000, " mm")
        self._spin_Ly = self._spin(0, 100000, 3000, " mm")
        self._spin_Lb = self._spin(0, 100000, 3000, " mm")
        self._spin_Kx = self._spin(0.1, 5, 1.0, "", decimals=2)
        self._spin_Ky = self._spin(0.1, 5, 1.0, "", decimals=2)
        gf.addRow("Lx:", self._spin_Lx)
        gf.addRow("Ly:", self._spin_Ly)
        gf.addRow("Lb (pandeo lat.):", self._spin_Lb)
        gf.addRow("Kx:", self._spin_Kx)
        gf.addRow("Ky:", self._spin_Ky)
        lv.addWidget(geom_grp)

        dem_grp = QGroupBox("Demandas (factorizadas)")
        dem_grp.setStyleSheet(_GRP)
        df = QFormLayout(dem_grp)
        df.setSpacing(6)

        self._spin_Pu = self._spin(-99999, 99999, 0, " kN")
        self._spin_Mux = self._spin(0, 99999, 0, " kN·m")
        self._spin_Muy = self._spin(0, 99999, 0, " kN·m")
        self._spin_Vux = self._spin(0, 99999, 0, " kN")
        df.addRow("Pu (+ tracción):", self._spin_Pu)
        df.addRow("Mux:", self._spin_Mux)
        df.addRow("Muy:", self._spin_Muy)
        df.addRow("Vux:", self._spin_Vux)
        lv.addWidget(dem_grp)

        tor_grp = QGroupBox("Torsión §H3")
        tor_grp.setStyleSheet(_GRP)
        tf = QFormLayout(tor_grp)
        tf.setSpacing(6)
        self._spin_Tu = self._spin(0, 99999, 0, " kN·m")
        self._spin_L_tor = self._spin(0, 100000, 3000, " mm")
        tf.addRow("Tu:", self._spin_Tu)
        tf.addRow("L (torsión):", self._spin_L_tor)
        lv.addWidget(tor_grp)

        self._spin_Cb = self._spin(1.0, 3.0, 1.0, "", decimals=2)
        cb_grp = QGroupBox("Flexión")
        cb_grp.setStyleSheet(_GRP)
        cbf = QFormLayout(cb_grp)
        cbf.setSpacing(6)
        cbf.addRow("Cb:", self._spin_Cb)
        lv.addWidget(cb_grp)

        # calculate button
        self._btn_calc = QPushButton("  Calcular AISC 360-22")
        self._btn_calc.setStyleSheet(_BTN)
        self._btn_calc.setMinimumHeight(38)
        self._btn_calc.setEnabled(False)
        self._btn_calc.clicked.connect(self._compute)
        lv.addWidget(self._btn_calc)

        self._status = StatusIndicator("info", "Sin perfil")
        lv.addWidget(self._status)
        lv.addStretch()
        return tab

    def _build_tab_sap(self) -> QWidget:
        tab = QWidget()
        lv = QVBoxLayout(tab)
        lv.setContentsMargins(8, 8, 8, 8)
        lv.setSpacing(8)

        lbl = QLabel("Integración SAP2000")
        lbl.setFont(QFont("Segoe UI", 11, QFont.Bold))
        lbl.setStyleSheet(f"color: {TEXT_PRIMARY};")
        lv.addWidget(lbl)

        try:
            from steeldesigner.sap2000.sap2000_worker import Sap2000Worker
            from steeldesigner.sap2000.sap2000_oapi import Sap2000Connector
            self._sap_available = True
        except ImportError:
            self._sap_available = False

        if not self._sap_available:
            lv.addWidget(QLabel("Módulo SAP2000 no disponible\nen este entorno."))
            lv.addStretch()
            return tab

        conn_grp = QGroupBox("Conexión")
        conn_grp.setStyleSheet(_GRP)
        cv = QVBoxLayout(conn_grp)
        self._progid_combo = QComboBox()
        self._progid_combo.addItems([
            "CSI.SAP2000.API.SapObject.23",
            "CSI.SAP2000.API.SapObject",
            "CSI.SAP2000.API.SapObject.22",
            "Fake (test)",
        ])
        cv.addWidget(self._progid_combo)
        self._btn_sap_connect = QPushButton("Conectar")
        self._btn_sap_connect.setStyleSheet(_BTN)
        self._btn_sap_connect.clicked.connect(self._toggle_sap)
        cv.addWidget(self._btn_sap_connect)
        self._sap_status = StatusIndicator("info", "Desconectado")
        cv.addWidget(self._sap_status)
        lv.addWidget(conn_grp)

        self._btn_sap_read = QPushButton("Leer selección SAP2000")
        self._btn_sap_read.setStyleSheet(_BTN2)
        self._btn_sap_read.setEnabled(False)
        self._btn_sap_read.clicked.connect(self._sap_read)
        lv.addWidget(self._btn_sap_read)

        self._btn_sap_write = QPushButton("Escribir K en SAP2000")
        self._btn_sap_write.setStyleSheet(_BTN2)
        self._btn_sap_write.setEnabled(False)
        self._btn_sap_write.clicked.connect(self._sap_write_k)
        lv.addWidget(self._btn_sap_write)

        self._sap_connector = None
        lv.addStretch()
        return tab

    # ── CENTER panel ─────────────────────────────────────────────────────────

    def _build_center(self) -> QWidget:
        w = QWidget()
        w.setMinimumWidth(240)
        w.setMaximumWidth(340)
        cv = QVBoxLayout(w)
        cv.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel("Vista del perfil")
        lbl.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:11px;")
        cv.addWidget(lbl)

        self._canvas = SectionCanvas()
        self._canvas.setMinimumHeight(240)
        cv.addWidget(self._canvas, stretch=2)

        self._detail_label = QLabel("—")
        self._detail_label.setWordWrap(True)
        self._detail_label.setAlignment(Qt.AlignTop)
        self._detail_label.setStyleSheet(f"font-size:11px; color:{TEXT_PRIMARY}; background:{BG_CARD}; padding:8px; border-radius:4px;")
        cv.addWidget(self._detail_label, stretch=3)
        return w

    # ── RIGHT panel ──────────────────────────────────────────────────────────

    def _build_right(self) -> QWidget:
        w = QWidget()
        rv = QVBoxLayout(w)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(8)

        # Metric cards row
        cards_row = QHBoxLayout()
        self._card_comp = MetricCard("φPn / Pn/Ω", "—", "Compresión")
        self._card_flex = MetricCard("φMnx / Mnx/Ω", "—", "Flexión mayor")
        self._card_shear = MetricCard("φVn / Vn/Ω", "—", "Cortante")
        self._card_dc = MetricCard("D/C", "—", "Interacción")
        for c in (self._card_comp, self._card_flex, self._card_shear, self._card_dc):
            cards_row.addWidget(c)
        rv.addLayout(cards_row)

        # Results table
        self._results_table = QTableWidget(0, 4)
        self._results_table.setHorizontalHeaderLabels(["Estado límite", "Ecuación", "φRn o Rn/Ω", "D/C"])
        self._results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, 4):
            self._results_table.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self._results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._results_table.setAlternatingRowColors(True)
        self._results_table.verticalHeader().setVisible(False)
        self._results_table.setStyleSheet(_TABLE)
        rv.addWidget(self._results_table, stretch=1)

        # HTML report tab
        tabs = QTabWidget()
        self._report_browser = QTextBrowser()
        self._report_browser.setMinimumHeight(200)
        tabs.addTab(self._report_browser, "Informe")
        rv.addWidget(tabs, stretch=1)

        # Export buttons
        btn_row = QHBoxLayout()
        self._btn_csv = QPushButton("Exportar CSV")
        self._btn_csv.setStyleSheet(_BTN2)
        self._btn_csv.setEnabled(False)
        self._btn_csv.clicked.connect(self._export_csv)
        btn_row.addWidget(self._btn_csv)

        self._btn_html = QPushButton("Guardar informe HTML")
        self._btn_html.setStyleSheet(_BTN2)
        self._btn_html.setEnabled(False)
        self._btn_html.clicked.connect(self._export_html)
        btn_row.addWidget(self._btn_html)
        btn_row.addStretch()
        rv.addLayout(btn_row)

        return w

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _spin(lo, hi, val, suffix, decimals=1) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        s.setDecimals(decimals)
        if suffix:
            s.setSuffix(suffix)
        return s

    def _show_section_info(self, sec):
        props = section_props_dict(sec)
        lines = [f"<b>{props['Designation']}</b>  <i>({props['Family']})</i>"]
        for k, v in props.items():
            if k in ("Designation", "Family"):
                continue
            lines.append(f"{k}: {v:,.1f}" if isinstance(v, float) else f"{k}: {v}")
        self._sec_info.setText("<br>".join(lines[:10]))  # truncate for space
        self._detail_label.setText("<br>".join(lines))

    # ── compute ──────────────────────────────────────────────────────────────

    def _compute(self):
        if self._section is None:
            QMessageBox.warning(self, "Sin perfil", "Seleccione un perfil del catálogo primero.")
            return

        method = "LRFD" if self._combo_method.currentIndex() == 0 else "ASD"
        inputs = DesignInputs(
            Fy=self._spin_Fy.value(),
            Fu=self._spin_Fu.value(),
            Lx=self._spin_Lx.value(),
            Ly=self._spin_Ly.value(),
            Lb=self._spin_Lb.value(),
            Kx=self._spin_Kx.value(),
            Ky=self._spin_Ky.value(),
            Pu=self._spin_Pu.value() * 1000,       # kN → N
            Mux=self._spin_Mux.value() * 1e6,       # kN·m → N·mm
            Muy=self._spin_Muy.value() * 1e6,
            Vux=self._spin_Vux.value() * 1000,      # kN → N
            Tu_torsion=self._spin_Tu.value() * 1e6, # kN·m → N·mm
            L_torsion=self._spin_L_tor.value(),
            Cb=self._spin_Cb.value(),
            method=method,
        )

        try:
            result = _FACADE.run(self._section, inputs)
            self._last_result = result
            self._populate_results(result)
            self._status.set_status(
                "ok" if result.interaction_ratio <= 1.0 else "error",
                f"D/C = {result.interaction_ratio:.3f}"
            )
            self._btn_csv.setEnabled(True)
            self._btn_html.setEnabled(True)
        except Exception as exc:
            self._status.set_status("error", f"Error: {exc}")
            QMessageBox.critical(self, "Error de cálculo", str(exc))

    def _populate_results(self, result):
        # Metric cards
        def fmt_cap(bundle, attr):
            if bundle is None:
                return "—"
            cap = getattr(bundle, "phi_Rn", None) or getattr(bundle, "capacity", None)
            if cap is None:
                return "—"
            return f"{cap/1000:.1f} kN" if "n" not in attr else f"{cap/1e6:.1f} kN·m"

        comp_cap = "—"
        flex_cap = "—"
        shear_cap = "—"

        if result.compression:
            v = getattr(result.compression, "phi_Pn", None) or getattr(result.compression, "phi_Rn", None)
            if v:
                comp_cap = f"{v/1000:.1f} kN"
        if result.flexure_major:
            v = getattr(result.flexure_major, "phi_Mn", None) or getattr(result.flexure_major, "phi_Rn", None)
            if v:
                flex_cap = f"{v/1e6:.1f} kN·m"
        if result.shear:
            v = getattr(result.shear, "phi_Vn", None) or getattr(result.shear, "phi_Rn", None)
            if v:
                shear_cap = f"{v/1000:.1f} kN"

        self._card_comp.set_value(comp_cap)
        self._card_flex.set_value(flex_cap)
        self._card_shear.set_value(shear_cap)

        dc = result.interaction_ratio
        self._card_dc.set_value(f"{dc:.3f}")
        color = "#16A34A" if dc <= 1.0 else "#DC2626"
        self._card_dc.set_color(color)

        # Results table
        self._results_table.setRowCount(0)
        rows = self._build_rows(result)
        for desc, eq, cap_str, ratio in rows:
            r = self._results_table.rowCount()
            self._results_table.insertRow(r)
            self._results_table.setItem(r, 0, QTableWidgetItem(desc))
            self._results_table.setItem(r, 1, QTableWidgetItem(eq))
            self._results_table.setItem(r, 2, QTableWidgetItem(cap_str))
            ratio_item = QTableWidgetItem(f"{ratio:.3f}" if ratio is not None else "—")
            if ratio is not None:
                ratio_item.setForeground(
                    Qt.green if ratio <= 1.0 else Qt.red
                )
            self._results_table.setItem(r, 3, ratio_item)

        # HTML report
        html = self._build_html(result)
        self._report_browser.setHtml(html)

    def _build_rows(self, result):
        rows = []

        def _from_bundle(bundle, label, unit_div=1000, unit_str="kN"):
            if bundle is None:
                return
            # Try to extract limit states from the bundle
            states = getattr(bundle, "limit_states", None) or getattr(bundle, "states", None)
            if states:
                for ls in states:
                    cap = getattr(ls, "phi_Rn", None) or getattr(ls, "capacity", None) or getattr(ls, "phi_Tn", None)
                    ratio = getattr(ls, "ratio", None)
                    cap_str = f"{cap/unit_div:.2f} {unit_str}" if cap else "—"
                    rows.append((label + ": " + getattr(ls, "description", ""), getattr(ls, "equation", "—"), cap_str, ratio))
            else:
                cap = getattr(bundle, "phi_Pn", None) or getattr(bundle, "phi_Mn", None) or \
                      getattr(bundle, "phi_Vn", None) or getattr(bundle, "phi_Rn", None)
                ratio = getattr(bundle, "ratio", None) or getattr(bundle, "dc_ratio", None)
                cap_str = f"{cap/unit_div:.2f} {unit_str}" if cap else "—"
                rows.append((label, "—", cap_str, ratio))

        _from_bundle(result.compression, "Compresión", 1000, "kN")
        _from_bundle(result.tension, "Tracción", 1000, "kN")
        _from_bundle(result.flexure_major, "Flexión Mx", 1e6, "kN·m")
        _from_bundle(result.flexure_minor, "Flexión My", 1e6, "kN·m")
        _from_bundle(result.shear, "Cortante", 1000, "kN")

        if result.torsion:
            for ls in result.torsion.states:
                cap_str = f"{ls.phi_Tn/1e6:.2f} kN·m"
                rows.append((f"Torsión: {ls.description}", ls.equation, cap_str, ls.ratio))

        return rows

    def _build_html(self, result) -> str:
        sec = self._section
        name = sec.designation_modern or sec.designation_legacy if sec else "?"
        ftype = result.family_type
        dc = result.interaction_ratio
        color = "#16A34A" if dc <= 1.0 else "#DC2626"

        rows_html = ""
        for desc, eq, cap_str, ratio in self._build_rows(result):
            ratio_str = f"{ratio:.3f}" if ratio is not None else "—"
            ok = ratio is not None and ratio <= 1.0
            rows_html += f"<tr><td>{desc}</td><td>{eq}</td><td>{cap_str}</td><td style='color:{'green' if ok else 'red'};font-weight:600'>{ratio_str}</td></tr>"

        torsion_notes = ""
        if result.torsion and result.torsion.notes:
            torsion_notes = "<p><small>" + "<br>".join(result.torsion.notes) + "</small></p>"

        error_html = f"<p style='color:red'><b>Error:</b> {result.error}</p>" if result.error else ""

        return f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<style>
body {{font-family:Segoe UI,Arial,sans-serif; font-size:12px; padding:12px; color:#1C1C1F;}}
h2 {{font-size:16px; margin:0 0 4px;}}
h3 {{font-size:13px; margin:8px 0 4px; color:#444;}}
table {{border-collapse:collapse; width:100%; margin:6px 0;}}
th,td {{border:1px solid #E0DDD8; padding:4px 8px; text-align:left;}}
th {{background:#F5F5F5; font-size:11px;}}
.badge {{display:inline-block; padding:3px 10px; border-radius:12px; color:white; font-weight:700; background:{color};}}
</style></head><body>
<h2>{name} <small style='color:#888'>({ftype})</small></h2>
<p>Método: {result.method} &nbsp;|&nbsp; D/C global: <span class='badge'>{dc:.3f}</span></p>
{error_html}
<h3>Verificaciones AISC 360-22</h3>
<table><tr><th>Estado límite</th><th>Ecuación</th><th>Capacidad</th><th>D/C</th></tr>
{rows_html}
</table>
{torsion_notes}
<p style='color:#888;font-size:10px'>Generado por SteelDesigner AISC 360-22</p>
</body></html>"""

    # ── export ───────────────────────────────────────────────────────────────

    def _export_csv(self):
        if not self._last_result:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Guardar CSV", "resultado.csv", "CSV (*.csv)")
        if not path:
            return
        rows = self._build_rows(self._last_result)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Estado límite", "Ecuación", "Capacidad", "D/C"])
            for row in rows:
                w.writerow(row)

    def _export_html(self):
        if not self._last_result:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Guardar HTML", "informe.html", "HTML (*.html)")
        if not path:
            return
        Path(path).write_text(self._build_html(self._last_result), encoding="utf-8")

    # ── SAP2000 stubs ─────────────────────────────────────────────────────────

    def _toggle_sap(self):
        if not self._sap_available:
            return
        try:
            from steeldesigner.sap2000.sap2000_oapi import Sap2000Connector
            if self._sap_connector is None:
                progid = self._progid_combo.currentText().split()[0]
                self._sap_connector = Sap2000Connector(progid)
                self._sap_connector.connect()
                self._btn_sap_connect.setText("Desconectar")
                self._sap_status.set_status("ok", "Conectado")
                self._btn_sap_read.setEnabled(True)
            else:
                self._sap_connector = None
                self._btn_sap_connect.setText("Conectar")
                self._sap_status.set_status("info", "Desconectado")
                self._btn_sap_read.setEnabled(False)
                self._btn_sap_write.setEnabled(False)
        except Exception as exc:
            self._sap_status.set_status("error", str(exc))

    def _sap_read(self):
        pass  # placeholder — full SAP2000 reading logic from SAP2000_Angles

    def _sap_write_k(self):
        pass  # placeholder
