"""
Página de catálogo de secciones de acero.

Zona izquierda: búsqueda + filtros de familia.
Zona central: tabla de resultados.
Zona derecha: dibujo + propiedades.
Doble-clic → emite señal section_selected(Section) para abrir en Diseño.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QPushButton, QLabel, QComboBox,
    QGroupBox, QSplitter, QAbstractItemView, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont

from steeldesigner.ui.theme import (
    BRAND, BG_SURFACE, BG_CARD, TEXT_PRIMARY, TEXT_SECONDARY, BORDER,
)
from steeldesigner.ui.widgets.section_canvas import SectionCanvas


_FAMILIES = [
    ("Todas", None),
    ("I soldado chileno (IN/HN/IP)", ["IN", "HN", "IP", "IE", "H", "PH", "HR"]),
    ("Wide-flange AISC (W/HP)", ["W", "HP"]),
    ("IPE / HEA / HEB / HEM europeo", ["IPE", "IPN", "HEA", "HEB", "HEM", "HL", "HD"]),
    ("Canal (C/MC/UPN)", ["C", "MC", "CA", "UPN", "UPE"]),
    ("Ángulo (L)", ["L_ICHA_LAM", "L_ICHA_PLEG", "L_AISC"]),
    ("Tee (WT/MT/ST)", ["WT", "MT", "ST", "T"]),
    ("HSS rectangular / cajón", ["CJ", "CJE", "HSS_R", "OC", "OCA"]),
    ("HSS circular / tubo", ["HSS_C", "O"]),
]

_COLS = ["Designación", "Familia", "d (mm)", "bf (mm)", "A (mm²)", "Peso (kg/m)", "Ix (mm⁴)", "Iy (mm⁴)"]


class CataloguePage(QWidget):
    section_selected = Signal(object)   # emite Section al hacer doble-clic

    def __init__(self, catalog, parent=None):
        super().__init__(parent)
        self._catalog = catalog
        self._results = []
        self._search_timer = QTimer()
        self._search_timer.setSingleShot(True)
        self._search_timer.timeout.connect(self._do_search)
        self._build_ui()
        self._do_search()

    # ------------------------------------------------------------------
    # Construcción de UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # --- Panel izquierdo: búsqueda + filtros ---
        left = QWidget()
        left.setMaximumWidth(280)
        left.setMinimumWidth(220)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel("Catálogo de Secciones")
        lbl.setFont(QFont("Segoe UI", 12, QFont.Bold))
        lbl.setStyleSheet(f"color: {TEXT_PRIMARY};")
        lv.addWidget(lbl)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Buscar perfil… (ej: W200, IN 300, L100)")
        self._search.textChanged.connect(self._on_text_changed)
        lv.addWidget(self._search)

        fam_box = QGroupBox("Familia")
        fam_box.setStyleSheet(f"QGroupBox {{ color: {TEXT_SECONDARY}; font-size: 11px; border: 1px solid {BORDER}; border-radius:4px; margin-top:6px; }} QGroupBox::title {{ subcontrol-origin: margin; left: 8px; }}")
        fv = QVBoxLayout(fam_box)
        fv.setContentsMargins(6, 12, 6, 6)
        self._family_combo = QComboBox()
        for label, _ in _FAMILIES:
            self._family_combo.addItem(label)
        self._family_combo.currentIndexChanged.connect(self._do_search)
        fv.addWidget(self._family_combo)
        lv.addWidget(fam_box)

        self._count_label = QLabel("0 perfiles")
        self._count_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
        lv.addWidget(self._count_label)

        self._open_btn = QPushButton("Diseñar perfil seleccionado →")
        self._open_btn.setEnabled(False)
        self._open_btn.clicked.connect(self._emit_selected)
        self._open_btn.setStyleSheet(
            f"QPushButton {{ background:{BRAND}; color:white; border:none; border-radius:6px; padding:8px 12px; font-weight:600; }}"
            f"QPushButton:disabled {{ background:{BORDER}; color:{TEXT_SECONDARY}; }}"
            f"QPushButton:hover {{ background:#0051A8; }}"
        )
        lv.addWidget(self._open_btn)
        lv.addStretch()
        splitter.addWidget(left)

        # --- Tabla central ---
        center = QWidget()
        cv = QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0)
        self._table = QTableWidget(0, len(_COLS))
        self._table.setHorizontalHeaderLabels(_COLS)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, len(_COLS)):
            self._table.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.doubleClicked.connect(self._on_double_click)
        self._table.selectionModel().selectionChanged.connect(self._on_selection)
        self._table.setStyleSheet(
            f"QTableWidget {{ font-size: 12px; gridline-color: {BORDER}; }}"
            f"QTableWidget::item:selected {{ background: #DBEAFE; color: {TEXT_PRIMARY}; }}"
        )
        cv.addWidget(self._table)
        splitter.addWidget(center)

        # --- Panel derecho: canvas + propiedades ---
        right = QWidget()
        right.setMinimumWidth(260)
        right.setMaximumWidth(320)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)

        self._canvas = SectionCanvas()
        self._canvas.setMinimumHeight(200)
        rv.addWidget(self._canvas, stretch=1)

        self._props_label = QLabel("—")
        self._props_label.setWordWrap(True)
        self._props_label.setAlignment(Qt.AlignTop)
        self._props_label.setStyleSheet(f"font-size:11px; color:{TEXT_PRIMARY}; background:{BG_CARD}; padding:8px; border-radius:4px;")
        rv.addWidget(self._props_label, stretch=1)
        splitter.addWidget(right)

        splitter.setSizes([240, 600, 280])

    # ------------------------------------------------------------------
    # Lógica
    # ------------------------------------------------------------------
    def _on_text_changed(self):
        self._search_timer.start(250)

    def _do_search(self):
        query = self._search.text().strip()
        idx = self._family_combo.currentIndex()
        families = _FAMILIES[idx][1]

        try:
            if query:
                results = self._catalog.fts_search(query, limit=300)
                if families:
                    results = [s for s in results
                               if s.family and s.family.family_code in families]
            else:
                kw = {"limit": 300}
                if families:
                    kw["family"] = families
                results = self._catalog.search(**kw)
        except Exception as e:
            results = []

        self._results = results
        self._populate_table(results)

    def _populate_table(self, sections):
        self._table.setRowCount(0)
        for sec in sections:
            row = self._table.rowCount()
            self._table.insertRow(row)
            fam_code = sec.family.family_code if sec.family else ""
            vals = [
                sec.designation_modern or sec.designation_legacy or "?",
                fam_code,
                f"{sec.d:.1f}" if sec.d else "—",
                f"{sec.bf:.1f}" if sec.bf else "—",
                f"{sec.area_mm2:.1f}" if sec.area_mm2 else "—",
                f"{sec.weight_kg_m:.2f}" if sec.weight_kg_m else "—",
                f"{sec.Ix_mm4:.0f}" if sec.Ix_mm4 else "—",
                f"{sec.Iy_mm4:.0f}" if sec.Iy_mm4 else "—",
            ]
            for col, val in enumerate(vals):
                item = QTableWidgetItem(str(val))
                item.setData(Qt.UserRole, sec)
                self._table.setItem(row, col, item)

        self._count_label.setText(f"{len(sections)} perfiles encontrados")
        self._open_btn.setEnabled(False)
        self._canvas.clear()
        self._props_label.setText("—")

    def _on_selection(self):
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            self._open_btn.setEnabled(False)
            return
        item = self._table.item(rows[0].row(), 0)
        if not item:
            return
        sec = item.data(Qt.UserRole)
        self._canvas.set_section(sec)
        self._show_props(sec)
        self._open_btn.setEnabled(True)

    def _on_double_click(self, index):
        item = self._table.item(index.row(), 0)
        if item:
            sec = item.data(Qt.UserRole)
            self.section_selected.emit(sec)

    def _emit_selected(self):
        rows = self._table.selectionModel().selectedRows()
        if rows:
            item = self._table.item(rows[0].row(), 0)
            if item:
                self.section_selected.emit(item.data(Qt.UserRole))

    def _show_props(self, sec):
        def fmt(v, unit=""):
            if v is None: return "—"
            return f"{v:,.1f} {unit}".strip()

        lines = [
            f"<b>{sec.designation_modern or sec.designation_legacy}</b>",
            f"Familia: {sec.family.family_code if sec.family else '—'}",
            f"d = {fmt(sec.d, 'mm')} &nbsp; bf = {fmt(sec.bf, 'mm')}",
            f"tf = {fmt(sec.tf, 'mm')} &nbsp; tw = {fmt(sec.tw, 'mm')}",
            f"A = {fmt(sec.area_mm2, 'mm²')}",
            f"Peso = {fmt(sec.weight_kg_m, 'kg/m')}",
            f"Ix = {fmt(sec.Ix_mm4, 'mm⁴')}",
            f"Sx = {fmt(sec.Sx_mm3, 'mm³')}",
            f"Zx = {fmt(sec.Zx_mm3, 'mm³')}",
            f"rx = {fmt(sec.rx_mm, 'mm')}",
            f"Iy = {fmt(sec.Iy_mm4, 'mm⁴')}",
            f"ry = {fmt(sec.ry_mm, 'mm')}",
            f"J = {fmt(sec.J_mm4, 'mm⁴')}",
            f"Cw = {fmt(sec.Cw_mm6, 'mm⁶')}",
        ]
        self._props_label.setText("<br>".join(lines))
