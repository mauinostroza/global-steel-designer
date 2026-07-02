"""
Entry point de SteelDesigner.

Configura el catálogo (desde variable de entorno o ruta relativa al ejecutable)
y lanza la aplicación PySide6.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _resolve_catalog_db() -> str:
    """Resuelve la ruta de catalog.db compatible con dev y .exe Nuitka standalone."""
    # 1. Variable de entorno explícita
    env = os.environ.get("SUITE_CATALOG_PATH")
    if env and Path(env).exists():
        return env

    # 2. Nuitka standalone: __file__ apunta al directorio de la app
    #    El archivo se incluye como steeldesigner/resources/catalog.db
    candidates = [
        Path(__file__).parent / "resources" / "catalog.db",
        Path(sys.argv[0]).parent / "steeldesigner" / "resources" / "catalog.db",
        Path(sys.argv[0]).parent / "resources" / "catalog.db",
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    raise FileNotFoundError(
        "No se encontró catalog.db. Configure la variable de entorno SUITE_CATALOG_PATH "
        f"o coloque el archivo en {candidates[0]}"
    )


def main():
    from PySide6.QtWidgets import QApplication, QMessageBox
    from PySide6.QtCore import Qt

    app = QApplication(sys.argv)
    app.setApplicationName("SteelDesigner")
    app.setOrganizationName("SuiteEstructural")
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # Cargar hoja de estilos global
    try:
        from steeldesigner.ui.theme import build_app_stylesheet
        app.setStyleSheet(build_app_stylesheet())
    except Exception:
        pass

    # Inicializar catálogo
    try:
        db_path = _resolve_catalog_db()
        from steeldesigner.catalog.catalog import Catalog
        catalog = Catalog.at_path(db_path)
    except FileNotFoundError as exc:
        # Mostrar error sin Qt si Qt no arrancó aún
        try:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Error — Catálogo no encontrado")
            msg.setText(str(exc))
            msg.exec()
        except Exception:
            print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR al cargar el catálogo: {exc}", file=sys.stderr)
        sys.exit(1)

    from steeldesigner.ui.main_window import MainWindow
    window = MainWindow(catalog)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
