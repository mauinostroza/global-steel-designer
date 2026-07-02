"""
Facade principal del catálogo de secciones.

Expone `Catalog.shared()` (singleton) y `Catalog.at_path()` (para tests),
y delega al `Repository` para todas las operaciones de datos.

Las apps consumidoras importan directamente desde aquí:
    import suite_steel_catalog as sc
    catalog = sc.Catalog.shared()
    sec = catalog.get("IN 200x100")
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Optional

from .models import Family, Material, Section
from .repository import Repository, SectionNotFoundError
from .migrations.runner import run_migrations


class CatalogNotFoundError(RuntimeError):
    """No se encontró catalog.db o está corrupto."""


class Catalog:
    """Facade singleton para acceder al catálogo de secciones.

    Uso típico:
        catalog = Catalog.shared()
        sec = catalog.get("IN 200x100")

    Para tests:
        catalog = Catalog.at_path("/tmp/test_catalog.db")
    """

    _shared: Optional["Catalog"] = None
    _shared_lock = threading.Lock()
    _shared_path: Optional[str] = None

    def __init__(self, db_path: str | Path, run_migrate: bool = True) -> None:
        self._db_path = str(db_path)
        self._repo = Repository(self._db_path)
        self._closed = False
        if run_migrate:
            self._ensure_migrations()

    # ------------------------------------------------------------------
    # Constructores de clase
    # ------------------------------------------------------------------
    @classmethod
    def shared(cls) -> "Catalog":
        """Retorna el singleton compartido.

        Busca catalog.db en este orden:
        1. Variable de entorno `SUITE_CATALOG_PATH`.
        2. %APPDATA%/SuiteEstructural/catalog.db (Windows)
           o ~/.local/share/SuiteEstructural/catalog.db (Linux/Mac).

        Lanza CatalogNotFoundError si no existe ningún archivo válido.
        """
        with cls._shared_lock:
            if cls._shared is not None and not cls._shared._closed:
                return cls._shared

            path = cls._shared_path or cls._resolve_default_path()
            if not Path(path).exists():
                raise CatalogNotFoundError(
                    f"Catálogo no encontrado en {path!r}. "
                    "Importe un paquete de actualización desde la app o "
                    "configure la variable de entorno SUITE_CATALOG_PATH."
                )

            cls._shared = cls(path)
            return cls._shared

    @classmethod
    def at_path(cls, db_path: str | Path, run_migrate: bool = True) -> "Catalog":
        """Crea una instancia apuntando a una ruta específica.

        Útil para tests y para desarrollo. No usa el singleton.
        """
        return cls(db_path, run_migrate=run_migrate)

    @classmethod
    def set_shared_path(cls, path: str | Path) -> None:
        """Override de la ruta del singleton (para tests / dev)."""
        cls._shared_path = str(path)
        # Invalidar el singleton existente para que la próxima llamada
        # a shared() use la nueva ruta.
        with cls._shared_lock:
            if cls._shared is not None:
                cls._shared.close()
                cls._shared = None

    @classmethod
    def reset_shared(cls) -> None:
        """Resetea el singleton (para tests)."""
        with cls._shared_lock:
            if cls._shared is not None:
                cls._shared.close()
                cls._shared = None
            cls._shared_path = None

    # ------------------------------------------------------------------
    # Resolución de ruta por defecto
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_default_path() -> str:
        # 1. Variable de entorno
        env_path = os.environ.get("SUITE_CATALOG_PATH")
        if env_path:
            return env_path

        # 2. Ruta por plataforma
        if sys.platform == "win32":
            appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
            base = Path(appdata) / "SuiteEstructural"
        else:
            base = Path.home() / ".local" / "share" / "SuiteEstructural"

        return str(base / "catalog.db")

    # ------------------------------------------------------------------
    # Migraciones
    # ------------------------------------------------------------------
    def _ensure_migrations(self) -> None:
        """Aplica migraciones pendientes al iniciar."""
        migrations_dir = Path(__file__).parent / "migrations"
        run_migrations(self._db_path, migrations_dir=migrations_dir, conn=self._repo.conn)

    # ------------------------------------------------------------------
    # API de búsqueda (delegada a Repository)
    # ------------------------------------------------------------------
    def get(self, designation: str) -> Section:
        """Búsqueda exacta por designación.

        Lanza SectionNotFoundError si no existe.
        """
        sec = self._repo.get(designation)
        # Carga perezosa de materiales
        self._repo.load_materials_for_section(sec)
        return sec

    def get_by_id(self, section_id: int) -> Section:
        sec = self._repo.get_by_id(section_id)
        self._repo.load_materials_for_section(sec)
        return sec

    def search(self, **filters) -> list[Section]:
        """Búsqueda por filtros. Ver Repository.search para parámetros."""
        results = self._repo.search(**filters)
        for sec in results:
            self._repo.load_materials_for_section(sec)
        return results

    def fuzzy_search(self, query: str, max_results: int = 20,
                     max_distance: Optional[int] = None) -> list[Section]:
        return self._repo.fuzzy_search(query, max_results=max_results,
                                        max_distance=max_distance)

    def fts_search(self, query: str, limit: int = 50) -> list[Section]:
        return self._repo.fts_search(query, limit=limit)

    # ------------------------------------------------------------------
    # Catálogos estáticos
    # ------------------------------------------------------------------
    def list_families(self) -> list[Family]:
        return self._repo.list_families()

    def get_family(self, family_code: str) -> Optional[Family]:
        return self._repo.get_family(family_code)

    def list_materials(self) -> list[Material]:
        return self._repo.list_materials()

    def get_material(self, material_code: str) -> Optional[Material]:
        return self._repo.get_material(material_code)

    def get_representative_section(self, family_code: str) -> Optional[Section]:
        """Perfil de tamaño mediano de la familia, útil para dibujar su ícono."""
        sec = self._repo.get_representative_section(family_code)
        if sec is not None:
            self._repo.load_materials_for_section(sec)
        return sec

    # ------------------------------------------------------------------
    # Conteos y métricas
    # ------------------------------------------------------------------
    def count_sections(self) -> int:
        return self._repo.count_sections()

    def count_sections_by_family(self) -> dict[str, int]:
        return self._repo.count_sections_by_family()

    def count_sections_by_source(self) -> dict[str, int]:
        return self._repo.count_sections_by_source()

    # ------------------------------------------------------------------
    # CRUD para perfiles custom (Fase 7, pero dejamos los hooks acá)
    # ------------------------------------------------------------------
    def create_custom(self, section: Section) -> int:
        """Crea un perfil custom. Retorna el section_id asignado."""
        return self._repo.create_custom(section)

    def delete_custom(self, section_id: int) -> None:
        """Elimina un perfil custom (solo si is_custom=True)."""
        self._repo.delete_custom(section_id)

    # ------------------------------------------------------------------
    # Versión
    # ------------------------------------------------------------------
    def version(self) -> str:
        """Versión actual del catálogo ('1.0.0', '1.1.0', ...)."""
        return self._repo.current_version()

    @property
    def db_path(self) -> str:
        return self._db_path

    # ------------------------------------------------------------------
    # Bulk insert para parsers (Fase 2)
    # ------------------------------------------------------------------
    def bulk_insert_records(self, records: list) -> tuple[int, int, int]:
        """Inserta una lista de SectionRecord del parser.

        Returns:
            (inserted, skipped_duplicates, errors)
        """
        return self._repo.bulk_insert_records(records)

    def dedupe_cross_source(self) -> int:
        """Fusiona perfiles duplicados provenientes de fuentes distintas.

        Ver Repository.dedupe_cross_source para el detalle. Debe invocarse
        tras cada ingesta completa para que la regeneración del catálogo
        no reintroduzca duplicados cross-fuente.

        Returns:
            Cantidad de grupos duplicados fusionados.
        """
        return self._repo.dedupe_cross_source()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def invalidate_cache(self) -> None:
        self._repo.invalidate_cache()

    def close(self) -> None:
        if not self._closed:
            self._repo.close()
            self._closed = True
