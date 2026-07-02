"""
Runner de migraciones SQL para el catálogo de secciones.

Aplica los scripts `NNN_*.sql` en orden numérico dentro de una transacción.
Registra cada migración aplicada en la tabla `schema_meta` para no reaplicarla.

Uso:
    from suite_steel_catalog.migrations.runner import run_migrations
    run_migrations(db_path="/path/to/catalog.db")
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Optional


class MigrationError(RuntimeError):
    """Error al aplicar una migración."""


@dataclass(frozen=True, slots=True)
class Migration:
    """Representa un archivo de migración SQL."""
    number: int          # 1, 2, 3, ...
    filename: str        # '001_initial.sql'
    description: str     # 'initial'
    sql: str             # contenido del archivo

    @property
    def key(self) -> str:
        """Clave única para registrar la migración aplicada."""
        return f"migration_{self.number:03d}_{self.description}"


def _discover_migrations(migrations_dir: Optional[Path] = None) -> list[Migration]:
    """Descubre y ordena las migraciones disponibles.

    Si no se pasa `migrations_dir`, usa el directorio del paquete
    `suite_steel_catalog.migrations`.
    """
    if migrations_dir is None:
        # Importlib.resources para soportar empaquetado con PyInstaller
        try:
            files = resources.files("suite_steel_catalog.migrations")
            migrations_dir = Path(str(files))
        except (ModuleNotFoundError, AttributeError) as exc:
            raise MigrationError(
                "No se pudo localizar el directorio de migraciones del paquete."
            ) from exc

    if not migrations_dir.is_dir():
        raise MigrationError(f"Directorio de migraciones no existe: {migrations_dir}")

    pattern = re.compile(r"^(\d{3})_(.+)\.sql$")
    migrations: list[Migration] = []

    for sql_file in sorted(migrations_dir.glob("*.sql")):
        match = pattern.match(sql_file.name)
        if not match:
            continue
        number = int(match.group(1))
        description = match.group(2)
        sql = sql_file.read_text(encoding="utf-8")
        migrations.append(Migration(number=number, filename=sql_file.name,
                                     description=description, sql=sql))

    return migrations


def _ensure_schema_meta(conn: sqlite3.Connection) -> None:
    """Asegura que la tabla schema_meta exista antes de consultarla."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    conn.commit()


def _applied_keys(conn: sqlite3.Connection) -> set[str]:
    """Retorna el conjunto de claves de migraciones ya aplicadas."""
    _ensure_schema_meta(conn)
    rows = conn.execute("SELECT key FROM schema_meta WHERE key LIKE 'migration_%'").fetchall()
    return {row[0] for row in rows}


def run_migrations(db_path: str | Path,
                   migrations_dir: Optional[Path] = None,
                   conn: Optional[sqlite3.Connection] = None) -> list[Migration]:
    """Aplica todas las migraciones pendientes en orden.

    Args:
        db_path: Ruta al archivo SQLite. Ignorada si se pasa `conn`.
        migrations_dir: Directorio con archivos NNN_*.sql.
            Si None, usa el del paquete.
        conn: Conexión SQLite ya abierta (opcional). Si se pasa,
            se usa y NO se cierra. Útil para tests.

    Returns:
        Lista de migraciones aplicadas en esta corrida (puede ser vacía).

    Raises:
        MigrationError: si alguna migración falla. En ese caso se hace
            rollback de la migración fallida pero las anteriores quedan
            aplicadas (cada una en su propia transacción).
    """
    migrations = _discover_migrations(migrations_dir)
    if not migrations:
        return []

    owns_conn = conn is None
    if owns_conn:
        conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        applied = _applied_keys(conn)
        newly_applied: list[Migration] = []

        for mig in migrations:
            if mig.key in applied:
                continue
            try:
                # Cada migración en su propia transacción.
                conn.execute("BEGIN")
                # executescript ejecuta múltiples sentencias SQL pero
                # NO soporta parámetros. Las migraciones son SQL estático,
                # así que es seguro.
                conn.executescript(mig.sql)
                conn.execute(
                    "INSERT INTO schema_meta (key, value) VALUES (?, ?)",
                    (mig.key, mig.filename),
                )
                conn.commit()
                newly_applied.append(mig)
            except sqlite3.Error as exc:
                conn.rollback()
                raise MigrationError(
                    f"Migración {mig.filename} fallida: {exc}"
                ) from exc

        return newly_applied
    finally:
        if owns_conn:
            conn.close()


def current_schema_version(conn: sqlite3.Connection) -> str:
    """Retorna la versión del esquema actual ('1.0.0', '1.1.0', ...)."""
    _ensure_schema_meta(conn)
    row = conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    ).fetchone()
    return row[0] if row else "0.0.0"
