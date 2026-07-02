"""
Paleta de colores — AngleDesigner.
Basado en guía corporativa: Rojo #D41E1E, Carbón #1C1C1F, Antracita #252528,
Gris cálido #F5F4F2, Blanco tarjeta #FFFFFF, Borde #E0DDD8.
"""

# ── Paleta Principal ──────────────────────────
BRAND        = "#D41E1E"   # Rojo Empresa — acento
TOPBAR       = "#1C1C1F"   # Carbón — topbar
SIDEBAR      = "#252528"   # Antracita — sidebar
BG_CONTENT   = "#F5F4F2"   # Gris cálido — fondo contenido
BG_SURFACE   = "#F5F4F2"   # alias de BG_CONTENT (toolbar/statusbar)
BG_CARD      = "#FFFFFF"   # Blanco — tarjetas
BORDER       = "#E0DDD8"   # Borde sutil
TEXT_PRIMARY = "#1C1C1F"   # Texto principal
TEXT_SECONDARY = "#666662" # Texto secundario
TEXT_INVERSE = "#FFFFFF"   # Texto sobre fondo oscuro
TEXT_MUTED   = "#999996"   # Texto deshabilitado

# ── Sidebar ────────────────────────────────────
SIDEBAR_TEXT       = "#999999"
SIDEBAR_ACTIVE_BG  = "rgba(212,30,30,0.12)"
SIDEBAR_ACTIVE_TEXT = "#E85555"
SIDEBAR_ACTIVE_BORDER = BRAND  # 2.5px solid

# ── Topbar ─────────────────────────────────────
TOPBAR_BTN_BG     = "rgba(255,255,255,0.08)"
TOPBAR_BTN_BORDER = "rgba(255,255,255,0.15)"
TOPBAR_BTN_TEXT   = "#D41E1E"

# ── Estados ────────────────────────────────────
ERROR_BG   = "#FDE8E8"
ERROR_TEXT = "#A32D2D"
OK_BG      = "#EAF3DE"
OK_TEXT    = "#3B6D11"
WARN_BG    = "#FAEEDA"
WARN_TEXT  = "#854F0B"
INFO_BG    = "#E6F1FB"
INFO_TEXT  = "#185FA5"

# ── Tablas ─────────────────────────────────────
TABLE_HEADER_BG      = "#F0EFED"
TABLE_HEADER_TEXT    = "#444444"
TABLE_ROW_EVEN       = "#FFFFFF"
TABLE_ROW_ODD        = "#FAFAF8"
TABLE_BORDER         = "#F0EFED"
TABLE_HOVER          = "#FFF5F5"

# ── Radios ─────────────────────────────────────
RADIUS_SM = "4px"
RADIUS_MD = "8px"
RADIUS_LG = "12px"

# ── Tipografía ─────────────────────────────────
FONT_FAMILY_UI   = "'Segoe UI', 'Inter', 'IBM Plex Sans', system-ui, sans-serif"
FONT_FAMILY_MONO = "'Cascadia Code', 'IBM Plex Mono', 'Consolas', monospace"
FONT_SIZE_BASE   = "13px"
FONT_SIZE_SM     = "11px"
FONT_SIZE_LG     = "15px"
FONT_SIZE_XL     = "18px"
FONT_SIZE_XXL    = "24px"


# ── QSS Global ─────────────────────────────────
def global_stylesheet() -> str:
    """Stylesheet QSS completo para toda la aplicación."""
    return f"""
    /* ── Global ── */
    QWidget {{
        font-family: {FONT_FAMILY_UI};
        font-size: {FONT_SIZE_BASE};
        color: {TEXT_PRIMARY};
    }}

    /* ── Scrollbars ── */
    QScrollBar:vertical {{
        background: {BG_CONTENT};
        width: 8px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: #C0BFBB;
        border-radius: 4px;
        min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: #A0A09B;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}

    QScrollBar:horizontal {{
        background: {BG_CONTENT};
        height: 8px;
        margin: 0;
    }}
    QScrollBar::handle:horizontal {{
        background: #C0BFBB;
        border-radius: 4px;
        min-width: 30px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: #A0A09B;
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0;
    }}

    /* ── Inputs ── */
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        padding: 6px 10px;
        border: 1px solid {BORDER};
        border-radius: {RADIUS_MD};
        background: {BG_CARD};
        color: {TEXT_PRIMARY};
        font-size: {FONT_SIZE_BASE};
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border: 1px solid {BRAND};
    }}
    QComboBox::drop-down {{
        border: none;
        padding-right: 8px;
    }}
    QComboBox QAbstractItemView {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        selection-background-color: {SIDEBAR_ACTIVE_BG};
        selection-color: {SIDEBAR_ACTIVE_TEXT};
    }}

    /* ── Buttons ── */
    QPushButton {{
        padding: 7px 16px;
        border-radius: {RADIUS_MD};
        font-size: {FONT_SIZE_BASE};
        font-weight: 500;
    }}
    QPushButton#btnPrimary {{
        background: {BRAND};
        color: {TEXT_INVERSE};
        border: none;
    }}
    QPushButton#btnPrimary:hover {{
        background: #B81A1A;
    }}
    QPushButton#btnPrimary:pressed {{
        background: #9E1616;
    }}
    QPushButton#btnSecondary {{
        background: {BG_CARD};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
    }}
    QPushButton#btnSecondary:hover {{
        background: #F0EFED;
    }}
    QPushButton#btnSuccess {{
        background: {OK_BG};
        color: {OK_TEXT};
        border: 1px solid {OK_TEXT};
    }}
    QPushButton#btnDanger {{
        background: {ERROR_BG};
        color: {ERROR_TEXT};
        border: 1px solid {ERROR_TEXT};
    }}

    /* ── Tooltips ── */
    QToolTip {{
        background: {TOPBAR};
        color: {TEXT_INVERSE};
        border: none;
        padding: 6px 10px;
        border-radius: {RADIUS_SM};
        font-size: {FONT_SIZE_SM};
    }}

    /* ── Tablas ── */
    QTableWidget {{
        border: 0.5px solid {BORDER};
        border-radius: {RADIUS_MD};
        background: {BG_CARD};
        gridline-color: {TABLE_BORDER};
        selection-background-color: {TABLE_HOVER};
        selection-color: {TEXT_PRIMARY};
    }}
    QTableWidget::item {{
        padding: 6px 10px;
    }}
    QHeaderView::section {{
        background: {TABLE_HEADER_BG};
        color: {TABLE_HEADER_TEXT};
        font-size: {FONT_SIZE_SM};
        font-weight: 500;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        padding: 8px 12px;
        border: none;
        border-bottom: 1px solid {BORDER};
    }}

    /* ── GroupBox ── */
    QGroupBox {{
        background: {BG_CARD};
        border: 0.5px solid {BORDER};
        border-radius: {RADIUS_MD};
        margin-top: 12px;
        padding: 16px 16px 12px 16px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 2px 10px;
        color: {TEXT_PRIMARY};
        font-weight: 500;
    }}

    /* ── Labels status ── */
    QLabel#statusOk {{
        background: {OK_BG};
        color: {OK_TEXT};
        padding: 4px 10px;
        border-radius: {RADIUS_SM};
        font-size: {FONT_SIZE_SM};
        font-weight: 500;
    }}
    QLabel#statusError {{
        background: {ERROR_BG};
        color: {ERROR_TEXT};
        padding: 4px 10px;
        border-radius: {RADIUS_SM};
        font-size: {FONT_SIZE_SM};
        font-weight: 500;
    }}
    QLabel#statusWarn {{
        background: {WARN_BG};
        color: {WARN_TEXT};
        padding: 4px 10px;
        border-radius: {RADIUS_SM};
        font-size: {FONT_SIZE_SM};
        font-weight: 500;
    }}
    QLabel#statusInfo {{
        background: {INFO_BG};
        color: {INFO_TEXT};
        padding: 4px 10px;
        border-radius: {RADIUS_SM};
        font-size: {FONT_SIZE_SM};
        font-weight: 500;
    }}

    /* ── ProgressBar ── */
    QProgressBar {{
        background: {TABLE_HEADER_BG};
        border: none;
        border-radius: {RADIUS_SM};
        height: 6px;
        text-align: center;
    }}
    QProgressBar::chunk {{
        background: {BRAND};
        border-radius: {RADIUS_SM};
    }}
    """
