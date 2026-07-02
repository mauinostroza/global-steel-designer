"""
Capa de acceso a datos del catálogo.

Encapsula todas las consultas SQLite y mantiene un caché LRU en memoria
para acelerar las búsquedas más frecuentes. Es la única capa que toca
sqlite3 directamente; ni la UI ni las apps consumidoras deben importar
sqlite3.

El caché LRU se invalida cuando:
- Se importa un paquete de actualización.
- Se crea/edita/elimina un perfil custom.
- Se cambia el material por defecto de un perfil.
"""
from __future__ import annotations

import re
import sqlite3
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

from .models import Family, Material, Section


class RepositoryError(RuntimeError):
    """Error genérico del repositorio."""


class SectionNotFoundError(KeyError):
    """Se buscó una designación que no existe en el catálogo."""

    def __init__(self, designation: str):
        self.designation = designation
        super().__init__(f"Perfil no encontrado: {designation!r}")


# SQL unificado para cargar una Section con todas sus propiedades.
# Los LEFT JOIN aseguran que las filas regresen incluso si la sección
# no tiene propiedades de torsión o pandeo local (sparse).
_SECTION_SELECT_SQL = """
    SELECT
        s.section_id, s.family_code, s.subclass,
        s.designation_modern, s.designation_legacy,
        s.designation_aisc, s.designation_en,
        s.source_catalog, s.source_edition,
        s.available_sack, s.available_cintac,
        s.is_standard, s.is_custom, s.notes,

        sd.d, sd.bf, sd.tf, sd.tw, sd.h, sd.T_clear,
        sd.k, sd.k_design, sd.k_detail,
        sd.r, sd.R_ext, sd.R_int,
        sd.B, sd.C_dim, sd.t_nom, sd.t_des,
        sd.area_mm2, sd.perimeter_mm, sd.weight_kg_m,
        sd.is_hollow, sd.is_sym_x, sd.is_sym_y,
        sd.centroid_x_mm, sd.centroid_y_mm,

        sps.Ix_mm4, sps.Sx_mm3, sps.Zx_mm3, sps.rx_mm, sps.xp_mm,
        spw.Iy_mm4, spw.Sy_mm3, spw.Zy_mm3, spw.ry_mm, spw.yp_mm,

        spt.J_mm4, spt.Cw_mm6, spt.Wno, spt.Sw, spt.Qf, spt.Qw,
        spt.H_const, spt.ro_mm, spt.xo_mm, spt.io_mm, spt.beta, spt.j,

        spl.bf_2tf, spl.h_tw, spl.b_t, spl.D_t,
        spl.Qs, spl.Qa, spl.ia, spl.it, spl.X1, spl.Fy_default_MPa
    FROM sections s
    LEFT JOIN section_dimensions sd ON sd.section_id = s.section_id
    LEFT JOIN section_properties_strong sps ON sps.section_id = s.section_id
    LEFT JOIN section_properties_weak spw ON spw.section_id = s.section_id
    LEFT JOIN section_properties_torsion spt ON spt.section_id = s.section_id
    LEFT JOIN section_properties_local_buckling spl ON spl.section_id = s.section_id
"""


class _LRUCache:
    """Caché LRU thread-safe para Section y Family.

    Las claves pueden ser ints (section_id) o strings (designación).
    El tamaño máximo es configurable; por defecto 500 entradas.
    """

    def __init__(self, max_size: int = 500) -> None:
        self._max = max_size
        self._data: OrderedDict[Any, Any] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: Any) -> Optional[Any]:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def put(self, key: Any, value: Any) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def invalidate(self, key: Optional[Any] = None) -> None:
        """Si key=None, vacía todo el caché."""
        with self._lock:
            if key is None:
                self._data.clear()
            else:
                self._data.pop(key, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


class Repository:
    """Capa de acceso a datos con caché LRU."""

    def __init__(self, db_path: str | Path, cache_size: int = 500) -> None:
        self._db_path = str(db_path)
        self._cache_section = _LRUCache(max_size=cache_size)
        self._cache_family: dict[str, Family] = {}
        self._cache_material: dict[str, Material] = {}
        self._cache_loaded = False
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Conexión
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        """Abre la conexión SQLite (lazy)."""
        if self._conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            try:
                conn.execute("PRAGMA journal_mode = WAL")
            except sqlite3.OperationalError:
                # WAL no soportado en algunos filesystems (p.ej. network shares)
                pass
            self._conn = conn
        return self._conn

    @property
    def conn(self) -> sqlite3.Connection:
        return self._connect()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
            self.invalidate_cache()

    # ------------------------------------------------------------------
    # Caché
    # ------------------------------------------------------------------
    def invalidate_cache(self) -> None:
        """Vacía todo el caché. Llamar tras importar/crear/editar/eliminar."""
        self._cache_section.invalidate()
        self._cache_family.clear()
        self._cache_material.clear()
        self._cache_loaded = False

    def _invalidate_section(self, section_id: int, designation: str) -> None:
        self._cache_section.invalidate(section_id)
        self._cache_section.invalidate(designation)
        self._cache_section.invalidate(designation.lower())

    # ------------------------------------------------------------------
    # Catálogos estáticos (families, materials)
    # ------------------------------------------------------------------
    def _ensure_static_cache(self) -> None:
        if self._cache_loaded:
            return
        with self._lock:
            if self._cache_loaded:
                return
            conn = self._connect()
            for row in conn.execute("SELECT * FROM families"):
                f = Family.from_row(row)
                self._cache_family[f.family_code] = f
            for row in conn.execute("SELECT * FROM materials"):
                m = Material.from_row(row)
                self._cache_material[m.material_code] = m
            self._cache_loaded = True

    def list_families(self) -> list[Family]:
        self._ensure_static_cache()
        return list(self._cache_family.values())

    def get_family(self, family_code: str) -> Optional[Family]:
        self._ensure_static_cache()
        return self._cache_family.get(family_code)

    def list_materials(self) -> list[Material]:
        self._ensure_static_cache()
        return list(self._cache_material.values())

    def get_material(self, material_code: str) -> Optional[Material]:
        self._ensure_static_cache()
        return self._cache_material.get(material_code)

    # ------------------------------------------------------------------
    # Búsqueda por designación (exacta)
    # ------------------------------------------------------------------
    def get(self, designation: str) -> Section:
        """Búsqueda exacta por designación (modern, legacy, AISC o EN).

        Lanza SectionNotFoundError si no existe.
        """
        # 1. Revisar caché por designación (case-insensitive)
        cached = self._cache_section.get(designation.lower())
        if cached is not None:
            return cached

        # 2. Buscar en BD por cualquiera de las 4 designaciones
        conn = self._connect()
        row = conn.execute(
            f"{_SECTION_SELECT_SQL} WHERE s.designation_modern = ? "
            f"   OR s.designation_legacy = ? "
            f"   OR s.designation_aisc = ? "
            f"   OR s.designation_en = ? "
            f"LIMIT 1",
            (designation, designation, designation, designation),
        ).fetchone()

        if row is None:
            raise SectionNotFoundError(designation)

        return self._row_to_section(row)

    def get_by_id(self, section_id: int) -> Section:
        """Búsqueda por section_id (clave primaria)."""
        cached = self._cache_section.get(section_id)
        if cached is not None:
            return cached
        conn = self._connect()
        row = conn.execute(
            f"{_SECTION_SELECT_SQL} WHERE s.section_id = ?",
            (section_id,),
        ).fetchone()
        if row is None:
            raise SectionNotFoundError(f"section_id={section_id}")
        return self._row_to_section(row)

    def _row_to_section(self, row: sqlite3.Row) -> Section:
        """Construye una Section desde una fila y la cachea."""
        family = self.get_family(row["family_code"])
        if family is None:
            raise RepositoryError(
                f"Familia {row['family_code']!r} no encontrada para section_id={row['section_id']}"
            )
        sec = Section.from_row(row, family)
        # Materiales perezosos: no se cargan aquí, se cargan bajo demanda.
        self._cache_section.put(sec.section_id, sec)
        self._cache_section.put(sec.designation_modern.lower(), sec)
        return sec

    # ------------------------------------------------------------------
    # Búsqueda por filtros
    # ------------------------------------------------------------------
    def search(self,
               family: Optional[str | list[str]] = None,
               subclass: Optional[str] = None,
               material: Optional[str] = None,
               source_catalog: Optional[str | list[str]] = None,
               d_min: Optional[float] = None,
               d_max: Optional[float] = None,
               weight_min: Optional[float] = None,
               weight_max: Optional[float] = None,
               available_sack: Optional[bool] = None,
               available_cintac: Optional[bool] = None,
               is_custom: Optional[bool] = None,
               limit: int = 200,
               offset: int = 0) -> list[Section]:
        """Búsqueda por filtros combinados.

        Todos los parámetros son opcionales y se combinan con AND.
        `family` y `source_catalog` pueden ser str o list[str] (OR dentro del campo).
        """
        where: list[str] = []
        params: list[Any] = []

        if family is not None:
            if isinstance(family, str):
                where.append("s.family_code = ?")
                params.append(family)
            else:
                placeholders = ",".join("?" * len(family))
                where.append(f"s.family_code IN ({placeholders})")
                params.extend(family)

        if subclass is not None:
            where.append("s.subclass = ?")
            params.append(subclass)

        if material is not None:
            where.append(
                "s.section_id IN (SELECT section_id FROM section_material_links "
                "WHERE material_code = ?)"
            )
            params.append(material)

        if source_catalog is not None:
            if isinstance(source_catalog, str):
                where.append("s.source_catalog = ?")
                params.append(source_catalog)
            else:
                placeholders = ",".join("?" * len(source_catalog))
                where.append(f"s.source_catalog IN ({placeholders})")
                params.extend(source_catalog)

        if d_min is not None:
            where.append("sd.d >= ?")
            params.append(d_min)
        if d_max is not None:
            where.append("sd.d <= ?")
            params.append(d_max)
        if weight_min is not None:
            where.append("sd.weight_kg_m >= ?")
            params.append(weight_min)
        if weight_max is not None:
            where.append("sd.weight_kg_m <= ?")
            params.append(weight_max)

        if available_sack is True:
            where.append("s.available_sack = 1")
        if available_cintac is True:
            where.append("s.available_cintac = 1")
        if is_custom is True:
            where.append("s.is_custom = 1")
        elif is_custom is False:
            where.append("s.is_custom = 0")

        where_clause = " AND ".join(where) if where else "1=1"
        sql = (
            f"{_SECTION_SELECT_SQL} WHERE {where_clause} "
            f"ORDER BY s.family_code, sd.d ASC, sd.weight_kg_m ASC "
            f"LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])

        conn = self._connect()
        rows = conn.execute(sql, params).fetchall()
        return [self._row_to_section(r) for r in rows]

    # ------------------------------------------------------------------
    # Búsqueda full-text (FTS5)
    # ------------------------------------------------------------------
    def fts_search(self, query: str, limit: int = 50) -> list[Section]:
        """Búsqueda full-text sobre las designaciones.

        Usa el índice FTS5 `sections_fts`. Acepta operadores FTS5:
        `AND`, `OR`, `NOT`, `*` (wildcard), `NEAR`.
        """
        if not query.strip():
            return []
        # Sanitizar: si el query no contiene operadores FTS5, agregar
        # wildcard al final para autocomplete.
        fts_ops = re.compile(r"\b(AND|OR|NOT|NEAR)\b", re.IGNORECASE)
        if not fts_ops.search(query) and "*" not in query and '"' not in query:
            fts_query = " OR ".join(
                f'"{tok}"*' for tok in query.split() if tok
            )
        else:
            fts_query = query

        conn = self._connect()
        try:
            rows = conn.execute(
                f"{_SECTION_SELECT_SQL} "
                f"JOIN sections_fts f ON f.section_id = s.section_id "
                f"WHERE sections_fts MATCH ? "
                f"ORDER BY f.rank LIMIT ?",
                (fts_query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            # FTS5 puede fallar con caracteres especiales. Caer a LIKE.
            like = f"%{query}%"
            rows = conn.execute(
                f"{_SECTION_SELECT_SQL} WHERE s.designation_modern LIKE ? "
                f"   OR s.designation_legacy LIKE ? "
                f"   OR s.designation_aisc LIKE ? "
                f"   OR s.designation_en LIKE ? "
                f"ORDER BY s.family_code, sd.d ASC, sd.weight_kg_m ASC LIMIT ?",
                (like, like, like, like, limit),
            ).fetchall()

        return [self._row_to_section(r) for r in rows]

    # ------------------------------------------------------------------
    # Búsqueda difusa (Levenshtein fallback)
    # ------------------------------------------------------------------
    def fuzzy_search(self, query: str, max_results: int = 20,
                     max_distance: Optional[int] = None) -> list[Section]:
        """Búsqueda difusa por designación.

        Implementación simple sin dependencia externa: si `python-Levenshtein`
        está disponible lo usa; si no, usa una implementación pura Python
        (más lenta pero suficiente para ~3 000 perfiles).
        """
        if not query.strip():
            return []

        # Primero intentar FTS exacto
        exact = self.fts_search(query, limit=max_results * 2)
        if exact:
            return exact[:max_results]

        # Si no hay resultados, recorrer todas las designaciones y comparar
        if max_distance is None:
            max_distance = 2 if len(query) < 8 else 3

        conn = self._connect()
        rows = conn.execute(
            "SELECT section_id, designation_modern, designation_legacy, "
            "       designation_aisc, designation_en "
            "FROM sections"
        ).fetchall()

        # Importar Levenshtein si está disponible; si no, usar implementación propia
        try:
            from Levenshtein import distance as _lev
        except ImportError:
            _lev = _levenshtein_pure

        query_lower = query.lower()
        scored: list[tuple[int, int, sqlite3.Row]] = []
        for r in rows:
            best_dist = 10_000
            for col in ("designation_modern", "designation_legacy",
                        "designation_aisc", "designation_en"):
                val = r[col]
                if val is None:
                    continue
                dist = _lev(query_lower, val.lower())
                if dist < best_dist:
                    best_dist = dist
            if best_dist <= max_distance:
                scored.append((best_dist, r["section_id"], r))

        scored.sort(key=lambda x: x[0])
        return [self.get_by_id(sid) for _, sid, _ in scored[:max_results]]

    # ------------------------------------------------------------------
    # Conteos
    # ------------------------------------------------------------------
    def count_sections(self) -> int:
        conn = self._connect()
        row = conn.execute("SELECT COUNT(*) FROM sections").fetchone()
        return int(row[0])

    def count_sections_by_family(self) -> dict[str, int]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT family_code, COUNT(*) AS n FROM sections GROUP BY family_code"
        ).fetchall()
        return {r["family_code"]: int(r["n"]) for r in rows}

    def count_sections_by_source(self) -> dict[str, int]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT source_catalog, COUNT(*) AS n FROM sections GROUP BY source_catalog"
        ).fetchall()
        return {r["source_catalog"]: int(r["n"]) for r in rows}

    # ------------------------------------------------------------------
    # Perfil representativo de una familia (para ícono en miniatura)
    # ------------------------------------------------------------------
    def get_representative_section(self, family_code: str) -> Optional[Section]:
        """Retorna un perfil "típico" (tamaño mediano por peso) de la
        familia, para usar como base del dibujo del ícono de esa familia.

        Toma la mediana de peso entre los perfiles estándar (no custom)
        de la familia; si no hay ninguno, retorna None.
        """
        conn = self._connect()
        row = conn.execute(
            "SELECT s.section_id FROM sections s "
            "JOIN section_dimensions sd ON sd.section_id = s.section_id "
            "WHERE s.family_code = ? AND s.is_custom = 0 AND sd.weight_kg_m IS NOT NULL "
            "ORDER BY sd.weight_kg_m "
            "LIMIT 1 OFFSET (SELECT COUNT(*) / 2 FROM sections s2 "
            "                JOIN section_dimensions sd2 ON sd2.section_id = s2.section_id "
            "                WHERE s2.family_code = ? AND s2.is_custom = 0 "
            "                  AND sd2.weight_kg_m IS NOT NULL)",
            (family_code, family_code),
        ).fetchone()
        if row is None:
            return None
        return self.get_by_id(row["section_id"])

    # ------------------------------------------------------------------
    # Deduplicación cross-fuente (migración 004, reutilizable en ingesta)
    # ------------------------------------------------------------------
    def dedupe_cross_source(self) -> int:
        """Fusiona perfiles duplicados que provienen de fuentes distintas
        pero comparten (family_code, subclass, designation_modern).

        Conserva la fila de mayor prioridad normativa (ICHA > Bechtel > EN
        > AISC > SACK > CINTAC > USER, empate por menor section_id), le
        funde encima available_sack/available_cintac y anota en `notes`
        qué fuentes se descartaron. Misma lógica que la migración SQL
        `004_dedupe_cross_source.sql`, expuesta aquí para poder invocarla
        después de cada regeneración completa del catálogo desde cero
        (ver `scripts/ingest_all.py`).

        Returns:
            Cantidad de grupos duplicados fusionados.
        """
        priority_case = (
            "CASE source_catalog "
            "WHEN 'ICHA' THEN 0 WHEN 'Bechtel' THEN 1 WHEN 'EN' THEN 2 "
            "WHEN 'AISC' THEN 3 WHEN 'SACK' THEN 4 WHEN 'CINTAC' THEN 5 "
            "ELSE 6 END"
        )
        conn = self._connect()
        conn.execute("BEGIN")
        try:
            conn.execute("DROP TABLE IF EXISTS _dup_ranked")
            conn.execute(f"""
                CREATE TEMP TABLE _dup_ranked AS
                SELECT
                    section_id, family_code,
                    COALESCE(subclass, '') AS subclass_key,
                    designation_modern, available_sack, available_cintac,
                    ROW_NUMBER() OVER (
                        PARTITION BY family_code, COALESCE(subclass, ''), designation_modern
                        ORDER BY {priority_case}, section_id
                    ) AS rn,
                    COUNT(*) OVER (
                        PARTITION BY family_code, COALESCE(subclass, ''), designation_modern
                    ) AS group_size
                FROM sections
            """)
            conn.execute("DROP TABLE IF EXISTS _dup_groups")
            conn.execute("""
                CREATE TEMP TABLE _dup_groups AS
                SELECT family_code, subclass_key, designation_modern,
                       MAX(CASE WHEN rn = 1 THEN section_id END) AS keeper_id,
                       MAX(available_sack) AS merged_available_sack,
                       MAX(available_cintac) AS merged_available_cintac
                FROM _dup_ranked
                WHERE group_size > 1
                GROUP BY family_code, subclass_key, designation_modern
            """)
            conn.execute("DROP TABLE _dup_ranked")
            n_groups = conn.execute("SELECT COUNT(*) FROM _dup_groups").fetchone()[0]
            if n_groups:
                conn.execute("""
                    UPDATE sections
                    SET available_sack = (
                            SELECT g.merged_available_sack FROM _dup_groups g
                            WHERE g.keeper_id = sections.section_id
                        ),
                        available_cintac = (
                            SELECT g.merged_available_cintac FROM _dup_groups g
                            WHERE g.keeper_id = sections.section_id
                        ),
                        notes = COALESCE(notes || ' | ', '') ||
                            'Fusionado con perfil(es) duplicado(s) de otra fuente: ' ||
                            (SELECT GROUP_CONCAT(s2.source_catalog || ' section_id=' || s2.section_id, ', ')
                             FROM sections s2
                             JOIN _dup_groups g2
                               ON g2.family_code = s2.family_code
                              AND g2.subclass_key = COALESCE(s2.subclass, '')
                              AND g2.designation_modern = s2.designation_modern
                             WHERE g2.keeper_id = sections.section_id
                               AND s2.section_id != sections.section_id)
                    WHERE section_id IN (SELECT keeper_id FROM _dup_groups)
                """)
                conn.execute("""
                    DELETE FROM sections
                    WHERE section_id IN (
                        SELECT s.section_id
                        FROM sections s
                        JOIN _dup_groups g
                          ON g.family_code = s.family_code
                         AND g.subclass_key = COALESCE(s.subclass, '')
                         AND g.designation_modern = s.designation_modern
                        WHERE s.section_id != g.keeper_id
                    )
                """)
            conn.execute("DROP TABLE _dup_groups")
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
            raise
        self.invalidate_cache()
        return n_groups

    # ------------------------------------------------------------------
    # Materiales de una sección (carga perezosa)
    # ------------------------------------------------------------------
    def load_materials_for_section(self, section: Section) -> None:
        """Carga la lista de materiales asociados a una sección.

        Es idempotente: si los materiales ya están cargados, no hace nada.
        """
        if section._materials_loaded:
            return
        conn = self._connect()
        rows = conn.execute(
            "SELECT m.*, sml.is_default FROM materials m "
            "JOIN section_material_links sml ON sml.material_code = m.material_code "
            "WHERE sml.section_id = ? "
            "ORDER BY sml.is_default DESC, m.material_code",
            (section.section_id,),
        ).fetchall()
        section.materials.clear()
        for row in rows:
            # row trae todos los campos de materials + is_default.
            # Construir Material con los primeros 8 campos.
            mat = Material(
                material_code=row["material_code"],
                standard=row["standard"],
                grade=row["grade"],
                Fy_MPa=float(row["Fy_MPa"]),
                Fu_MPa=float(row["Fu_MPa"]) if row["Fu_MPa"] is not None else None,
                E_MPa=float(row["E_MPa"]) if row["E_MPa"] is not None else 200_000.0,
                nu=float(row["nu"]) if row["nu"] is not None else 0.30,
                applicable_families=row["applicable_families"],
            )
            section.materials.append(mat)
            if row["is_default"]:
                section._default_material = mat
        section._materials_loaded = True

    # ------------------------------------------------------------------
    # CRUD para perfiles custom (Fase 7, pero dejamos los hooks acá)
    # ------------------------------------------------------------------
    def create_custom(self, section: Section) -> int:
        """Inserta un perfil custom. Retorna el section_id asignado."""
        if not section.is_custom:
            raise RepositoryError("Solo se pueden crear perfiles marcados como is_custom=True")
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "INSERT INTO sections (family_code, subclass, designation_modern, "
                    "  designation_legacy, designation_aisc, designation_en, source_catalog, "
                    "  source_edition, available_sack, available_cintac, is_standard, "
                    "  is_custom, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (section.family.family_code, section.subclass,
                     section.designation_modern, section.designation_legacy,
                     section.designation_aisc, section.designation_en,
                     "USER", section.source_edition,
                     int(section.available_sack), int(section.available_cintac),
                     0, 1, section.notes),
                )
                section_id = cur.lastrowid
                # Insertar dimensiones
                self._upsert_dimensions(conn, section_id, section)
                self._upsert_properties_strong(conn, section_id, section)
                self._upsert_properties_weak(conn, section_id, section)
                self._upsert_properties_torsion(conn, section_id, section)
                self._upsert_properties_local_buckling(conn, section_id, section)
                conn.commit()
                section.section_id = section_id
                self._invalidate_section(section_id, section.designation_modern)
                return section_id
            except sqlite3.Error as exc:
                conn.rollback()
                raise RepositoryError(f"No se pudo crear perfil custom: {exc}") from exc

    def _upsert_dimensions(self, conn: sqlite3.Connection,
                           section_id: int, s: Section) -> None:
        conn.execute(
            "INSERT INTO section_dimensions (section_id, d, bf, tf, tw, h, T_clear, "
            "  k, k_design, k_detail, r, R_ext, R_int, B, C_dim, t_nom, t_des, "
            "  area_mm2, perimeter_mm, weight_kg_m, is_hollow, is_sym_x, is_sym_y, "
            "  centroid_x_mm, centroid_y_mm) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (section_id, s.d, s.bf, s.tf, s.tw, s.h, s.T_clear,
             s.k, s.k_design, s.k_detail, s.r, s.R_ext, s.R_int,
             s.B, s.C_dim, s.t_nom, s.t_des,
             s.area_mm2, s.perimeter_mm, s.weight_kg_m,
             int(s.is_hollow), int(s.is_sym_x), int(s.is_sym_y),
             s.centroid_x_mm, s.centroid_y_mm),
        )

    def _upsert_properties_strong(self, conn: sqlite3.Connection,
                                  section_id: int, s: Section) -> None:
        conn.execute(
            "INSERT INTO section_properties_strong (section_id, Ix_mm4, Sx_mm3, "
            "  Zx_mm3, rx_mm, xp_mm) VALUES (?, ?, ?, ?, ?, ?)",
            (section_id, s.Ix_mm4, s.Sx_mm3, s.Zx_mm3, s.rx_mm, s.xp_mm),
        )

    def _upsert_properties_weak(self, conn: sqlite3.Connection,
                                section_id: int, s: Section) -> None:
        conn.execute(
            "INSERT INTO section_properties_weak (section_id, Iy_mm4, Sy_mm3, "
            "  Zy_mm3, ry_mm, yp_mm) VALUES (?, ?, ?, ?, ?, ?)",
            (section_id, s.Iy_mm4, s.Sy_mm3, s.Zy_mm3, s.ry_mm, s.yp_mm),
        )

    def _upsert_properties_torsion(self, conn: sqlite3.Connection,
                                   section_id: int, s: Section) -> None:
        if all(v is None for v in (s.J_mm4, s.Cw_mm6, s.Wno, s.Sw,
                                   s.Qf, s.Qw, s.H_const, s.ro_mm,
                                   s.xo_mm, s.io_mm, s.beta, s.j)):
            return
        conn.execute(
            "INSERT INTO section_properties_torsion (section_id, J_mm4, Cw_mm6, "
            "  Wno, Sw, Qf, Qw, H_const, ro_mm, xo_mm, io_mm, beta, j) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (section_id, s.J_mm4, s.Cw_mm6, s.Wno, s.Sw,
             s.Qf, s.Qw, s.H_const, s.ro_mm, s.xo_mm,
             s.io_mm, s.beta, s.j),
        )

    def _upsert_properties_local_buckling(self, conn: sqlite3.Connection,
                                          section_id: int, s: Section) -> None:
        if all(v is None for v in (s.bf_2tf, s.h_tw, s.b_t, s.D_t,
                                   s.Qs, s.Qa, s.ia, s.it, s.X1, s.Fy_default_MPa)):
            return
        conn.execute(
            "INSERT INTO section_properties_local_buckling (section_id, bf_2tf, h_tw, "
            "  b_t, D_t, Qs, Qa, ia, it, X1, Fy_default_MPa) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (section_id, s.bf_2tf, s.h_tw, s.b_t, s.D_t,
             s.Qs, s.Qa, s.ia, s.it, s.X1, s.Fy_default_MPa),
        )

    def delete_custom(self, section_id: int) -> None:
        """Elimina un perfil custom. Lanza PermissionError si no es custom."""
        conn = self._connect()
        row = conn.execute(
            "SELECT is_custom, designation_modern FROM sections WHERE section_id = ?",
            (section_id,),
        ).fetchone()
        if row is None:
            raise SectionNotFoundError(f"section_id={section_id}")
        if not row["is_custom"]:
            raise PermissionError(
                f"El perfil {row['designation_modern']!r} no es custom y no puede eliminarse"
            )
        # ON DELETE CASCADE en las FK elimina automáticamente todas las
        # tablas hijas (dimensions, properties_*, material_links).
        conn.execute("DELETE FROM sections WHERE section_id = ?", (section_id,))
        conn.commit()
        self._invalidate_section(section_id, row["designation_modern"])

    # ------------------------------------------------------------------
    # Versión del catálogo
    # ------------------------------------------------------------------
    def current_version(self) -> str:
        """Versión actual del catálogo (la más reciente por semver)."""
        conn = self._connect()
        # Usar ORDER BY rowid DESC para obtener la última insertada,
        # ya que varias migraciones pueden tener el mismo installed_at
        # (ej. ejecutadas en el mismo segundo).
        row = conn.execute(
            "SELECT version FROM catalog_versions ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        return row["version"] if row else "0.0.0"

    # ------------------------------------------------------------------
    # Bulk insert para parsers (Fase 2)
    # ------------------------------------------------------------------
    def bulk_insert_records(self, records: list) -> tuple[int, int, int]:
        """Inserta una lista de SectionRecord del parser.

        Usa un pre-check SELECT con COALESCE(subclass, '') para detectar
        duplicados de forma confiable incluso cuando subclass=NULL
        (SQLite trata NULL como distinto en UNIQUE constraints estándar,
        pero la migración 003 agrega un índice UNIQUE basado en expresión
        que trata NULL como '').

        Cada registro se inserta en su propio SAVEPOINT para que un error
        en uno no afecte a los demás.

        Returns:
            (inserted, skipped_duplicates, errors)
        """
        inserted = 0
        skipped = 0
        errors = 0
        conn = self._connect()
        try:
            conn.execute("BEGIN")
            for i, rec in enumerate(records):
                sp_name = f"sp_{i}"
                conn.execute(f"SAVEPOINT {sp_name}")
                try:
                    # Pre-check: ¿ya existe un perfil con la misma
                    # (family_code, subclass tratando NULL como '',
                    #  designation_modern, source_catalog)?
                    # Esto es necesario porque el constraint UNIQUE a
                    # nivel de tabla trata NULL como distinto, pero el
                    # índice único basado en expresión (migración 003)
                    # trata NULL como ''.
                    existing = conn.execute(
                        "SELECT section_id FROM sections "
                        "WHERE family_code = ? "
                        "  AND COALESCE(subclass, '') = COALESCE(?, '') "
                        "  AND designation_modern = ? "
                        "  AND source_catalog = ?",
                        (rec.family_code, rec.subclass,
                         rec.designation_modern, rec.source_catalog),
                    ).fetchone()
                    if existing is not None:
                        # Duplicado: rollback del savepoint y continuar
                        conn.execute(f"ROLLBACK TO {sp_name}")
                        conn.execute(f"RELEASE {sp_name}")
                        skipped += 1
                        continue
                    section_id = self._insert_section_row(conn, rec)
                    if section_id is None:
                        # El INSERT falló por otra razón (constraint,
                        # FK, etc.) — tratar como error
                        conn.execute(f"ROLLBACK TO {sp_name}")
                        conn.execute(f"RELEASE {sp_name}")
                        errors += 1
                        continue

                    # Dimensions
                    conn.execute(
                        "INSERT INTO section_dimensions (section_id, d, bf, tf, tw, "
                        "  h, T_clear, k, k_design, k_detail, r, R_ext, R_int, B, "
                        "  C_dim, t_nom, t_des, area_mm2, perimeter_mm, weight_kg_m, "
                        "  is_hollow, is_sym_x, is_sym_y, centroid_x_mm, centroid_y_mm) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (section_id, rec.d, rec.bf, rec.tf, rec.tw, rec.h, rec.T_clear,
                         rec.k, rec.k_design, rec.k_detail, rec.r, rec.R_ext, rec.R_int,
                         rec.B, rec.C_dim, rec.t_nom, rec.t_des,
                         rec.area_mm2, rec.perimeter_mm, rec.weight_kg_m,
                         int(rec.is_hollow), int(rec.is_sym_x), int(rec.is_sym_y),
                         rec.centroid_x_mm, rec.centroid_y_mm),
                    )

                    # Strong axis
                    if any(v is not None for v in (rec.Ix_mm4, rec.Sx_mm3, rec.Zx_mm3,
                                                    rec.rx_mm, rec.xp_mm)):
                        conn.execute(
                            "INSERT INTO section_properties_strong (section_id, Ix_mm4, "
                            "  Sx_mm3, Zx_mm3, rx_mm, xp_mm) VALUES (?, ?, ?, ?, ?, ?)",
                            (section_id, rec.Ix_mm4, rec.Sx_mm3, rec.Zx_mm3,
                             rec.rx_mm, rec.xp_mm),
                        )

                    # Weak axis
                    if any(v is not None for v in (rec.Iy_mm4, rec.Sy_mm3, rec.Zy_mm3,
                                                    rec.ry_mm, rec.yp_mm)):
                        conn.execute(
                            "INSERT INTO section_properties_weak (section_id, Iy_mm4, "
                            "  Sy_mm3, Zy_mm3, ry_mm, yp_mm) VALUES (?, ?, ?, ?, ?, ?)",
                            (section_id, rec.Iy_mm4, rec.Sy_mm3, rec.Zy_mm3,
                             rec.ry_mm, rec.yp_mm),
                        )

                    # Torsion
                    if any(v is not None for v in (rec.J_mm4, rec.Cw_mm6, rec.Wno,
                                                    rec.Sw, rec.Qf, rec.Qw,
                                                    rec.H_const, rec.ro_mm,
                                                    rec.xo_mm, rec.io_mm,
                                                    rec.beta, rec.j)):
                        conn.execute(
                            "INSERT INTO section_properties_torsion (section_id, J_mm4, "
                            "  Cw_mm6, Wno, Sw, Qf, Qw, H_const, ro_mm, xo_mm, io_mm, "
                            "  beta, j) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (section_id, rec.J_mm4, rec.Cw_mm6, rec.Wno, rec.Sw,
                             rec.Qf, rec.Qw, rec.H_const, rec.ro_mm, rec.xo_mm,
                             rec.io_mm, rec.beta, rec.j),
                        )

                    # Local buckling
                    if any(v is not None for v in (rec.bf_2tf, rec.h_tw, rec.b_t,
                                                    rec.D_t, rec.Qs, rec.Qa, rec.ia,
                                                    rec.it, rec.X1, rec.Fy_default_MPa)):
                        conn.execute(
                            "INSERT INTO section_properties_local_buckling (section_id, "
                            "  bf_2tf, h_tw, b_t, D_t, Qs, Qa, ia, it, X1, Fy_default_MPa) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (section_id, rec.bf_2tf, rec.h_tw, rec.b_t, rec.D_t,
                             rec.Qs, rec.Qa, rec.ia, rec.it, rec.X1, rec.Fy_default_MPa),
                        )

                    conn.execute(f"RELEASE {sp_name}")
                    inserted += 1
                except sqlite3.Error:
                    # Rollback al savepoint y continuar con el siguiente
                    try:
                        conn.execute(f"ROLLBACK TO {sp_name}")
                        conn.execute(f"RELEASE {sp_name}")
                    except sqlite3.Error:
                        # Si el rollback falla, la transacción está corrupta
                        # y debemos abortar todo
                        conn.rollback()
                        raise
                    errors += 1
                    continue
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self.invalidate_cache()
        return inserted, skipped, errors

    def _insert_section_row(self, conn: sqlite3.Connection, rec) -> Optional[int]:
        """Inserta una fila en sections y retorna el section_id.

        Returns None si el INSERT falló (ej. constraint violation).
        """
        try:
            cur = conn.execute(
                "INSERT INTO sections (family_code, subclass, "
                "  designation_modern, designation_legacy, designation_aisc, "
                "  designation_en, source_catalog, source_edition, "
                "  available_sack, available_cintac, is_standard, is_custom, "
                "  notes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (rec.family_code, rec.subclass, rec.designation_modern,
                 rec.designation_legacy, rec.designation_aisc,
                 rec.designation_en, rec.source_catalog, rec.source_edition,
                 int(rec.available_sack), int(rec.available_cintac),
                 1, 0, rec.notes),
            )
            return cur.lastrowid
        except sqlite3.Error:
            return None


# ----------------------------------------------------------------------
# Implementación pura Python de Levenshtein (fallback)
# ----------------------------------------------------------------------
def _levenshtein_pure(s1: str, s2: str) -> int:
    """Distancia de edición de Levenshtein (pura Python, sin dependencias)."""
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]
