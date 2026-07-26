"""Página Selector de Perfiles — diseño automático + ranking por peso y D/C.

Flujo:
  1. Usuario filtra familias / d_min-max / peso_max  → live-count de secciones
  2. Usuario ingresa cargas manualmente o importa desde SAP2000
  3. Click "Diseñar todos"  → ThreadPoolExecutor en QThread
  4. Tabla de ranking: perfiles que cumplen (D/C ≤ 1) ordenados por peso
  5. Doble-click en fila  → emite section_selected(Section)
"""
from __future__ import annotations

import csv
import os

from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QFont, QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar,
    QSplitter, QScrollArea, QFileDialog, QMessageBox, QSizePolicy,
)

from steeldesigner.catalog.models import Section
from steeldesigner.core.engine_facade import DesignInputs
from steeldesigner.core.profile_selector import SelectorCandidate, select_profiles
from steeldesigner.ui.theme import (
    BRAND, BG_CARD, BG_SURFACE, BG_CONTENT, TEXT_PRIMARY, TEXT_SECONDARY,
    BORDER, OK_BG, OK_TEXT, ERROR_BG, ERROR_TEXT, WARN_BG, WARN_TEXT,
    TABLE_HEADER_BG, TABLE_HEADER_TEXT, TABLE_ROW_EVEN, TABLE_ROW_ODD,
    TABLE_HOVER, RADIUS_MD,
)


# ── Familia groups (same as catalogue_page) ────────────────────────────────────

_FAMILY_GROUPS: list[tuple[str, list[str]]] = [
    ("I/H Americano (W, HP)",         ["W", "HP"]),
    ("I/H Europeo (IPE, HEA, HEB…)",  ["IPE", "IPN", "HEA", "HEB", "HEM", "HL", "HD"]),
    ("I soldado chileno (IN, HN, IP)", ["IN", "HN", "IP", "IE", "H", "PH", "HR"]),
    ("Canal (C, MC, UPN)",             ["C", "MC", "CA", "UPN", "UPE"]),
    ("Tubo rect. (CJ, CJE, HSS_R)",    ["CJ", "CJE", "HSS_R", "OC", "OCA"]),
    ("Tubo circ. (O, HSS_C)",          ["HSS_C", "O"]),
    ("Ángulo (L)",                      ["L_ICHA_LAM", "L_ICHA_PLEG", "L_AISC"]),
]

_RANK_COLS = ["#", "Designación", "Peso\n(kg/m)", "D/C", "Pasa",
              "Ix\n(cm⁴)", "A\n(cm²)"]


# ── Background worker ──────────────────────────────────────────────────────────

class _SelectorWorker(QObject):
    progress = Signal(int, int)          # (completados, total)
    finished = Signal(list)              # list[SelectorCandidate]
    failed = Signal(str)

    def __init__(self, sections: list[Section], inputs: DesignInputs):
        super().__init__()
        self._sections = sections
        self._inputs = inputs

    def run(self):
        try:
            results = select_profiles(
                self._sections,
                self._inputs,
                max_workers=4,
                progress_cb=lambda done, tot: self.progress.emit(done, tot),
            )
            self.finished.emit(results)
        except Exception as exc:
            self.failed.emit(str(exc))


# ── Main page widget ───────────────────────────────────────────────────────────

class SelectorPage(QWidget):
    section_selected = Signal(object)   # emite Section al hacer doble-clic

    def __init__(self, catalog, parent=None):
        super().__init__(parent)
        self._catalog = catalog
        self._candidates: list[SelectorCandidate] = []
        self._thread: QThread | None = None
        self._worker: _SelectorWorker | None = None
        self._sections_for_run: list[Section] = []
        self._build_ui()
        self._refresh_count()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter)

        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_center())
        splitter.addWidget(self._build_right())
        splitter.setSizes([250, 340, 600])

    # ── Left panel — filters ───────────────────────────────────────────────────

    def _build_left(self) -> QWidget:
        panel = QWidget()
        panel.setMaximumWidth(300)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        # Title
        title = QLabel("Filtro de perfiles")
        title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        lay.addWidget(title)

        # Family checkboxes
        fam_box = QGroupBox("Familias")
        fam_lay = QVBoxLayout(fam_box)
        fam_lay.setSpacing(4)
        self._family_checks: dict[str, tuple[QCheckBox, list[str]]] = {}
        for label, codes in _FAMILY_GROUPS:
            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.stateChanged.connect(self._refresh_count)
            self._family_checks[label] = (cb, codes)
            fam_lay.addWidget(cb)

        lay.addWidget(fam_box)

        # Dimension filters
        dim_box = QGroupBox("Dimensiones")
        dim_lay = QVBoxLayout(dim_box)

        dim_row1 = QHBoxLayout()
        dim_row1.addWidget(QLabel("d min (mm)"))
        self._d_min = QSpinBox()
        self._d_min.setRange(0, 2000)
        self._d_min.setValue(0)
        self._d_min.setSingleStep(10)
        self._d_min.valueChanged.connect(self._refresh_count)
        dim_row1.addWidget(self._d_min)
        dim_lay.addLayout(dim_row1)

        dim_row2 = QHBoxLayout()
        dim_row2.addWidget(QLabel("d max (mm)"))
        self._d_max = QSpinBox()
        self._d_max.setRange(0, 2000)
        self._d_max.setValue(2000)
        self._d_max.setSingleStep(10)
        self._d_max.valueChanged.connect(self._refresh_count)
        dim_row2.addWidget(self._d_max)
        dim_lay.addLayout(dim_row2)

        dim_row3 = QHBoxLayout()
        dim_row3.addWidget(QLabel("Peso max (kg/m)"))
        self._weight_max = QDoubleSpinBox()
        self._weight_max.setRange(0, 2000)
        self._weight_max.setValue(500)
        self._weight_max.setSingleStep(10)
        self._weight_max.valueChanged.connect(self._refresh_count)
        dim_row3.addWidget(self._weight_max)
        dim_lay.addLayout(dim_row3)

        lay.addWidget(dim_box)

        # Count label
        self._count_label = QLabel("0 perfiles seleccionados")
        self._count_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
        lay.addWidget(self._count_label)

        # Run button
        self._btn_run = QPushButton("▶  Diseñar todos")
        self._btn_run.setObjectName("btnPrimary")
        self._btn_run.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self._btn_run.setMinimumHeight(44)
        self._btn_run.clicked.connect(self._run_selector)
        lay.addWidget(self._btn_run)

        # Progress bar (hidden until running)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        self._progress.setFixedHeight(8)
        lay.addWidget(self._progress)

        lay.addStretch()
        return panel

    # ── Center panel — loads ───────────────────────────────────────────────────

    def _build_center(self) -> QWidget:
        panel = QWidget()
        panel.setMaximumWidth(380)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        title = QLabel("Cargas de diseño")
        title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        lay.addWidget(title)

        # SAP2000 import button
        sap_box = QGroupBox("SAP2000")
        sap_lay = QVBoxLayout(sap_box)
        self._btn_sap = QPushButton("⬇  Importar desde SAP2000")
        self._btn_sap.setObjectName("btnSecondary")
        self._btn_sap.clicked.connect(self._import_from_sap)
        sap_lay.addWidget(self._btn_sap)

        self._sap_combo = QComboBox()
        self._sap_combo.setVisible(False)
        self._sap_combo.currentIndexChanged.connect(self._apply_sap_selection)
        sap_lay.addWidget(self._sap_combo)

        self._sap_status = QLabel("")
        self._sap_status.setWordWrap(True)
        self._sap_status.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
        sap_lay.addWidget(self._sap_status)
        lay.addWidget(sap_box)

        # Loads input
        loads_box = QGroupBox("Demandas (LRFD factorizadas)")
        loads_lay = QVBoxLayout(loads_box)
        self._load_fields: dict[str, QDoubleSpinBox] = {}
        load_defs = [
            ("Pu (kN)",   "Pu",   -9999, 9999, 0.0, 1.0),
            ("Mux (kN·m)", "Mux",  0, 99999, 0.0, 1.0),
            ("Muy (kN·m)", "Muy",  0, 99999, 0.0, 0.1),
            ("Vux (kN)",   "Vux",  0, 9999,  0.0, 1.0),
        ]
        for lbl, key, lo, hi, default, step in load_defs:
            row = QHBoxLayout()
            row.addWidget(QLabel(lbl))
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setValue(default)
            sp.setSingleStep(step)
            sp.setDecimals(2)
            self._load_fields[key] = sp
            row.addWidget(sp)
            loads_lay.addLayout(row)
        lay.addWidget(loads_box)

        # Lengths
        len_box = QGroupBox("Longitudes y parámetros")
        len_lay = QVBoxLayout(len_box)
        self._len_fields: dict[str, QDoubleSpinBox] = {}
        len_defs = [
            ("Lx (m)",  "Lx",   0.1, 100.0, 3.0),
            ("Ly (m)",  "Ly",   0.1, 100.0, 3.0),
            ("Lb (m)",  "Lb",   0.1, 100.0, 3.0),
            ("Fy (MPa)", "Fy",  100, 700,   250.0),
            ("Cb",      "Cb",   1.0, 3.0,   1.0),
        ]
        for lbl, key, lo, hi, default in len_defs:
            row = QHBoxLayout()
            row.addWidget(QLabel(lbl))
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setValue(default)
            sp.setDecimals(3 if key == "Cb" else 1)
            self._len_fields[key] = sp
            row.addWidget(sp)
            len_lay.addLayout(row)

        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Método"))
        self._method_combo = QComboBox()
        self._method_combo.addItems(["LRFD", "ASD"])
        method_row.addWidget(self._method_combo)
        len_lay.addLayout(method_row)
        lay.addWidget(len_box)

        lay.addStretch()
        return panel

    # ── Right panel — ranking table ────────────────────────────────────────────

    def _build_right(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        hdr = QHBoxLayout()
        self._result_title = QLabel("Ranking de perfiles")
        self._result_title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        hdr.addWidget(self._result_title)
        hdr.addStretch()
        self._btn_export = QPushButton("Exportar CSV")
        self._btn_export.setObjectName("btnSecondary")
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self._export_csv)
        hdr.addWidget(self._btn_export)
        lay.addLayout(hdr)

        self._table = QTableWidget(0, len(_RANK_COLS))
        self._table.setHorizontalHeaderLabels(_RANK_COLS)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        for col in [0, 2, 3, 4, 5, 6]:
            self._table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self._table.doubleClicked.connect(self._on_row_dblclick)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(f"""
            QTableWidget::item:alternate {{ background: {TABLE_ROW_ODD}; }}
        """)
        lay.addWidget(self._table)

        return panel

    # ── Slot: refresh section count ────────────────────────────────────────────

    def _refresh_count(self):
        families = self._active_families()
        if not families:
            self._count_label.setText("0 perfiles seleccionados")
            return
        d_min = self._d_min.value() or None
        d_max = self._d_max.value() if self._d_max.value() < 2000 else None
        w_max = self._weight_max.value() if self._weight_max.value() < 500 else None

        kwargs: dict = {"family": families, "limit": 1000}
        if d_min:
            kwargs["d_min"] = float(d_min)
        if d_max:
            kwargs["d_max"] = float(d_max)
        if w_max:
            kwargs["weight_max"] = float(w_max)

        secs = self._catalog.search(**kwargs)
        self._count_label.setText(f"{len(secs)} perfiles seleccionados")

    def _active_families(self) -> list[str]:
        codes: list[str] = []
        for cb, family_codes in self._family_checks.values():
            if cb.isChecked():
                codes.extend(family_codes)
        return codes

    # ── Slot: run selector ─────────────────────────────────────────────────────

    def _run_selector(self):
        families = self._active_families()
        if not families:
            QMessageBox.warning(self, "Sin familias", "Selecciona al menos una familia de perfiles.")
            return

        d_min = self._d_min.value() or None
        d_max = self._d_max.value() if self._d_max.value() < 2000 else None
        w_max = self._weight_max.value() if self._weight_max.value() < 500 else None

        kwargs: dict = {"family": families, "limit": 1000}
        if d_min:
            kwargs["d_min"] = float(d_min)
        if d_max:
            kwargs["d_max"] = float(d_max)
        if w_max:
            kwargs["weight_max"] = float(w_max)

        sections = self._catalog.search(**kwargs)
        if not sections:
            QMessageBox.information(self, "Sin perfiles", "No se encontraron perfiles con los filtros actuales.")
            return

        inputs = self._build_inputs()

        self._sections_for_run = sections
        self._btn_run.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._table.setRowCount(0)
        self._result_title.setText(f"Calculando {len(sections)} perfiles…")

        self._thread = QThread()
        self._worker = _SelectorWorker(sections, inputs)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.start()

    def _build_inputs(self) -> DesignInputs:
        f = self._load_fields
        l = self._len_fields
        Pu = f["Pu"].value() * 1000.0    # kN → N
        Mux = f["Mux"].value() * 1e6     # kN·m → N·mm
        Muy = f["Muy"].value() * 1e6
        Vux = f["Vux"].value() * 1000.0
        Lx = l["Lx"].value() * 1000.0   # m → mm
        Ly = l["Ly"].value() * 1000.0
        Lb = l["Lb"].value() * 1000.0
        Fy = l["Fy"].value()
        Cb = l["Cb"].value()
        method = self._method_combo.currentText()

        # Pu negative = compresión en AISC convention
        return DesignInputs(
            Fy=Fy,
            E=200_000.0,
            Lx=Lx, Ly=Ly, Lb=Lb,
            Pu=max(0.0, Pu),           # compresión
            Tu_axial=max(0.0, -Pu),    # tracción
            Mux=Mux, Muy=Muy,
            Vux=Vux,
            Cb=Cb,
            method=method,
        )

    # ── Worker signals ─────────────────────────────────────────────────────────

    def _on_progress(self, done: int, total: int):
        pct = int(done * 100 / total) if total else 0
        self._progress.setValue(pct)

    def _on_finished(self, candidates: list[SelectorCandidate]):
        self._candidates = candidates
        self._populate_table(candidates)
        self._progress.setVisible(False)
        self._btn_run.setEnabled(True)
        passing = sum(1 for c in candidates if c.result.passes_interaction and not c.result.error)
        self._result_title.setText(
            f"Ranking — {passing} cumplen de {len(candidates)} evaluados"
        )
        self._btn_export.setEnabled(bool(candidates))

    def _on_failed(self, msg: str):
        self._progress.setVisible(False)
        self._btn_run.setEnabled(True)
        self._result_title.setText("Error en cálculo")
        QMessageBox.critical(self, "Error", f"Error durante el cálculo:\n{msg}")

    def _cleanup_thread(self):
        self._thread = None
        self._worker = None

    # ── Table population ───────────────────────────────────────────────────────

    def _populate_table(self, candidates: list[SelectorCandidate]):
        self._table.setRowCount(0)
        self._table.setRowCount(len(candidates))

        for row, cand in enumerate(candidates):
            sec = cand.section
            res = cand.result
            passes = res.passes_interaction and not res.error
            ratio = res.interaction_ratio

            Ix_cm4 = (sec.Ix_mm4 or 0.0) / 1e4
            A_cm2 = (sec.area_mm2 or 0.0) / 100.0

            cells = [
                (str(cand.rank),                    Qt.AlignCenter),
                (sec.designation_modern or "—",     Qt.AlignLeft | Qt.AlignVCenter),
                (f"{sec.weight_kg_m:.1f}",           Qt.AlignCenter),
                (f"{ratio:.3f}",                     Qt.AlignCenter),
                ("✓" if passes else "✗",             Qt.AlignCenter),
                (f"{Ix_cm4:.0f}" if Ix_cm4 else "—", Qt.AlignCenter),
                (f"{A_cm2:.1f}" if A_cm2 else "—",   Qt.AlignCenter),
            ]

            for col, (text, align) in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(align)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

                if col == 3:  # D/C
                    if res.error:
                        item.setForeground(QColor(TEXT_SECONDARY))
                    elif passes and ratio >= 0.85:
                        item.setForeground(QColor(OK_TEXT))
                        item.setBackground(QColor(OK_BG))
                    elif passes:
                        item.setForeground(QColor(TEXT_PRIMARY))
                    else:
                        item.setForeground(QColor(ERROR_TEXT))
                        item.setBackground(QColor(ERROR_BG))

                if col == 4:  # Pasa
                    if passes:
                        item.setForeground(QColor(OK_TEXT))
                    else:
                        item.setForeground(QColor(ERROR_TEXT))

                # Store Section in column 1 for retrieval on double-click
                if col == 1:
                    item.setData(Qt.UserRole, sec)

                self._table.setItem(row, col, item)

    # ── Double-click → open in DesignPage ─────────────────────────────────────

    def _on_row_dblclick(self, index):
        item = self._table.item(index.row(), 1)
        if item:
            sec = item.data(Qt.UserRole)
            if sec:
                self.section_selected.emit(sec)

    # ── SAP2000 import ─────────────────────────────────────────────────────────

    def _import_from_sap(self):
        try:
            from steeldesigner.sap2000.sap2000_oapi import Sap2000Connector
            from steeldesigner.sap2000.load_reader import read_selected_frame_loads
        except ImportError as exc:
            self._sap_status.setText(f"Error importando módulo SAP2000: {exc}")
            return

        try:
            connector = Sap2000Connector()
            connector.connect()
        except Exception as exc:
            self._sap_status.setText(f"No se pudo conectar a SAP2000:\n{exc}")
            return

        try:
            loads_list = read_selected_frame_loads(connector)
        except Exception as exc:
            self._sap_status.setText(f"Error leyendo fuerzas:\n{exc}")
            return

        if not loads_list:
            self._sap_status.setText("No se encontraron elementos seleccionados o sin resultados.")
            return

        self._sap_loads = loads_list
        self._sap_combo.clear()
        for fl in loads_list:
            self._sap_combo.addItem(f"{fl.element_name} — {fl.combo_name}")
        self._sap_combo.setVisible(len(loads_list) > 1)

        # Apply first element immediately
        self._apply_sap_loads(loads_list[0])
        total = len(loads_list)
        self._sap_status.setText(
            f"✓ {total} elemento(s) leídos desde SAP2000."
            + (f" Selecciona uno en el desplegable." if total > 1 else "")
        )

    def _apply_sap_selection(self, idx: int):
        if hasattr(self, "_sap_loads") and 0 <= idx < len(self._sap_loads):
            self._apply_sap_loads(self._sap_loads[idx])

    def _apply_sap_loads(self, fl):
        from steeldesigner.sap2000.load_reader import FrameLoads
        self._load_fields["Pu"].setValue(fl.Pu_N / 1000.0)
        self._load_fields["Mux"].setValue(fl.Mux_Nmm / 1e6)
        self._load_fields["Muy"].setValue(fl.Muy_Nmm / 1e6)
        self._load_fields["Vux"].setValue(fl.Vux_N / 1000.0)
        if fl.length_mm > 0:
            L_m = fl.length_mm / 1000.0
            self._len_fields["Lx"].setValue(L_m)
            self._len_fields["Ly"].setValue(L_m)
            self._len_fields["Lb"].setValue(L_m)

    # ── CSV export ─────────────────────────────────────────────────────────────

    def _export_csv(self):
        if not self._candidates:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar ranking", "ranking_perfiles.csv",
            "CSV (*.csv)"
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["Rank", "Designación", "Peso (kg/m)", "D/C",
                                  "Pasa", "Ix (cm4)", "A (cm2)", "Error"])
                for c in self._candidates:
                    sec = c.section
                    res = c.result
                    Ix = (sec.Ix_mm4 or 0) / 1e4
                    A = (sec.area_mm2 or 0) / 100
                    writer.writerow([
                        c.rank,
                        sec.designation_modern,
                        f"{sec.weight_kg_m:.2f}",
                        f"{res.interaction_ratio:.4f}",
                        "SI" if res.passes_interaction and not res.error else "NO",
                        f"{Ix:.1f}",
                        f"{A:.2f}",
                        res.error or "",
                    ])
            QMessageBox.information(self, "Exportado", f"Archivo guardado:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"No se pudo guardar:\n{exc}")
