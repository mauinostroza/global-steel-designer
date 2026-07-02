"""
SteelDesigner Nuitka Build Script.
Genera EXE standalone portable para Windows.

Uso:
    python build.py             # build standalone
    python build.py --onefile   # todo en un solo .exe
"""

import os
import sys
import subprocess
import platform


def build(onefile: bool = False):
    if platform.system() != "Windows":
        print("El build Nuitka solo funciona en Windows.")
        print("  En Linux/Mac ejecuta: python -m steeldesigner.app")
        return

    root = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(root, "src")
    icon = os.path.join(src, "steeldesigner", "resources", "logo.ico")
    catalog_db = os.path.join(src, "steeldesigner", "resources", "catalog.db")
    out = os.path.join(root, "dist")
    os.makedirs(out, exist_ok=True)

    cmd = [
        sys.executable, "-m", "nuitka",
        "--standalone",
        "--enable-plugin=pyside6",
        "--windows-console-mode=disable",
        "--assume-yes-for-downloads",
        f"--output-dir={out}",
        "--include-package=steeldesigner",
        "--include-package-data=steeldesigner:resources",
        # Incluir catalog.db explícitamente en la ruta correcta del EXE
        f"--include-data-files={catalog_db}=steeldesigner/resources/catalog.db",
        "--output-filename=SteelDesigner",
    ]

    if os.path.exists(icon):
        cmd.append(f"--windows-icon-from-ico={icon}")

    if onefile:
        cmd.append("--onefile")

    app_script = os.path.join(src, "steeldesigner", "app.py")
    cmd.append(app_script)

    print("=" * 60)
    print("  SteelDesigner -- Nuitka Build")
    print("=" * 60)
    print(f"  SOURCE:  {app_script}")
    print(f"  OUTPUT:  {out}")
    print(f"  MODE:    {'onefile' if onefile else 'standalone'}")
    print(f"  DB:      {'OK' if os.path.exists(catalog_db) else 'NO ENCONTRADO'}")
    print("=" * 60)

    try:
        subprocess.run(cmd, check=True, cwd=root)
        print("\nBuild completado. Busca el EXE en: dist/SteelDesigner.dist/")
    except subprocess.CalledProcessError as e:
        print(f"Build falló con código {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print("Nuitka no encontrado. Instala con: pip install nuitka")
        sys.exit(1)


if __name__ == "__main__":
    onefile = "--onefile" in sys.argv
    build(onefile=onefile)
