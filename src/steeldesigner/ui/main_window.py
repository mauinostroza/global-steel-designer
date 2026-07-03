"""
Ventana principal — QMainWindow con barra de herramientas y QStackedWidget.

Páginas:
  0 — Catálogo de secciones
  1 — Diseño AISC 360-22
  2 — Historial / resultados

Ctrl+1/2/3 navegan entre páginas.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QStackedWidget, QToolBar, QLabel,
    QStatusBar, QSizePolicy,
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QAction, QKeySequence, QFont

from steeldesigner.ui.theme import BRAND, BG_SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, BORDER
from steeldesigner.ui.pages.catalogue_page import CataloguePage
from steeldesigner.ui.pages.design_page import DesignPage
from steeldesigner.ui.pages.results_page import ResultsPage


class MainWindow(QMainWindow):
    def __init__(self, catalog, parent=None):
        super().__init__(parent)
        self._catalog = catalog
        self._results_history: list[dict] = []

        self.setWindowTitle("SteelDesigner — AISC 360-22")
        self.resize(1280, 800)
        self.setMinimumSize(900, 600)

        self._build_toolbar()
        self._build_pages()
        self._build_statusbar()

        # Start on catalogue
        self._switch(0)

    # ── toolbar ──────────────────────────────────────────────────────────────

    def _build_toolbar(self):
        tb = QToolBar("Navegación")
        tb.setMovable(False)
        tb.setIconSize(QSize(20, 20))
        tb.setStyleSheet(f"""
            QToolBar {{
                background: {BG_SURFACE};
                border-bottom: 1px solid {BORDER};
                padding: 4px 12px;
                spacing: 4px;
            }}
            QToolButton {{
                color: {TEXT_SECONDARY};
                font-size: 13px;
                padding: 6px 14px;
                border-radius: 6px;
                border: none;
            }}
            QToolButton:hover {{ background: #E8E8E8; color: {TEXT_PRIMARY}; }}
            QToolButton:checked {{ background: {BRAND}; color: white; font-weight: 600; }}
        """)

        title = QLabel("  SteelDesigner")
        title.setFont(QFont("Segoe UI", 13, QFont.Bold))
        title.setStyleSheet(f"color: {BRAND}; padding-right:16px;")
        tb.addWidget(title)

        self._act_catalogue = self._nav_action(tb, "Catálogo", "Ctrl+1", 0)
        self._act_design = self._nav_action(tb, "Diseño", "Ctrl+2", 1)
        self._act_results = self._nav_action(tb, "Resultados", "Ctrl+3", 2)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)

        version_lbl = QLabel("AISC 360-22  ")
        version_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
        tb.addWidget(version_lbl)

        self.addToolBar(tb)

    def _nav_action(self, tb: QToolBar, label: str, shortcut: str, page_idx: int) -> QAction:
        act = QAction(label, self)
        act.setShortcut(QKeySequence(shortcut))
        act.setCheckable(True)
        act.triggered.connect(lambda checked=False, i=page_idx: self._switch(i))
        tb.addAction(act)
        return act

    # ── pages ────────────────────────────────────────────────────────────────

    def _build_pages(self):
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._catalogue_page = CataloguePage(self._catalog)
        self._catalogue_page.section_selected.connect(self._open_in_design)
        self._stack.addWidget(self._catalogue_page)

        self._design_page = DesignPage()
        self._design_page.result_computed.connect(self._on_result_computed)
        self._stack.addWidget(self._design_page)

        self._results_page = ResultsPage()
        self._stack.addWidget(self._results_page)

    # ── status bar ───────────────────────────────────────────────────────────

    def _build_statusbar(self):
        sb = QStatusBar()
        sb.setStyleSheet(f"background: {BG_SURFACE}; color: {TEXT_SECONDARY}; font-size: 11px; border-top: 1px solid {BORDER};")
        self._status_msg = QLabel("Listo")
        sb.addWidget(self._status_msg)
        self.setStatusBar(sb)

    # ── navigation ───────────────────────────────────────────────────────────

    def _switch(self, idx: int):
        self._stack.setCurrentIndex(idx)
        for i, act in enumerate([self._act_catalogue, self._act_design, self._act_results]):
            act.setChecked(i == idx)

    def _open_in_design(self, section):
        self._design_page.load_section(section)
        self._switch(1)
        name = section.designation_modern or section.designation_legacy
        self._status_msg.setText(f"Perfil cargado: {name}")

    def _on_result_computed(self, result):
        """Convierte DesignResult al dict que ResultsPage.add_result() espera."""
        comp = result.compression
        phi_pn = getattr(comp, "phi_Pn", None)
        ratio = getattr(comp, "ratio", None) or result.interaction_ratio

        row = {
            "section": result.section_name,
            "family": result.family_type,
            "method": result.method,
            "Pu_kN": 0.0,
            "phiPn_kN": phi_pn / 1000.0 if phi_pn else 0.0,
            "ratio": ratio,
            "ok": result.interaction_ratio <= 1.0,
            "reserve_pct": max(0.0, (1.0 - result.interaction_ratio) * 100),
            "deficit_pct": max(0.0, (result.interaction_ratio - 1.0) * 100),
            "(KL/r)eff": result.geometry_source.get("angle_e5", {}).get("KLr_eff", 0.0) if result.geometry_source else 0.0,
            "(KL/r)design": result.geometry_source.get("angle_e5", {}).get("KLr_design", 0.0) if result.geometry_source else 0.0,
            "Fcr_MPa": result.geometry_source.get("angle_e5", {}).get("Fcr_MPa", 0.0) if result.geometry_source else 0.0,
            "mode": result.geometry_source.get("angle_e5", {}).get("mode", result.family_type) if result.geometry_source else result.family_type,
            "conn_leg": "—",
        }
        self._results_page.add_result(row)
        self._status_msg.setText(f"Calculado: {result.section_name} — D/C = {result.interaction_ratio:.3f}")
