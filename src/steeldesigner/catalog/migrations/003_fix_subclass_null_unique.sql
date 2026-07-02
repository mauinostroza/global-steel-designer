-- ============================================================
-- Migración 003: Corregir duplicados con subclass=NULL
-- Versión: 1.0.1
-- Fecha: 2026-06-21
-- ============================================================
-- Problema:
--   SQLite trata NULL como distinto en constraints UNIQUE, por lo que
--   filas con subclass=NULL pueden duplicarse incluso con el constraint
--   UNIQUE(family_code, subclass, designation_modern, source_catalog)
--   a nivel de tabla. Esto causaba ~50 duplicados en la BD (perfiles
--   Bechtel con subclass=NULL que aparecían dos veces).
--
-- Solución:
--   1. Deduplica filas existentes (conserva la de menor section_id,
--      que típicamente tiene los datos más completos por orden de
--      inserción ICHA → Bechtel).
--   2. Crea un índice UNIQUE basado en expresión que trata NULL como
--      '' (string vacío), previniendo futuros duplicados.
--
-- Notas:
--   - El constraint UNIQUE a nivel de tabla se conserva (no estorba;
--     permite NULL=NULL pero el índice nuevo es el que previene
--     duplicados de forma efectiva).
--   - La deduplicación usa ON DELETE CASCADE en las FK, así que las
--     tablas hijas (dimensions, properties_*, material_links) de las
--     filas eliminadas también se borran.
--   - El FTS5 se mantiene sincronizado automáticamente por los triggers.
-- ============================================================

PRAGMA foreign_keys = ON;

-- ------------------------------------------------------------
-- Step 1: Identificar y eliminar duplicados
-- ------------------------------------------------------------
-- Para cada grupo de duplicados (mismo family_code, subclass
-- tratando NULL como '', designation_modern, source_catalog),
-- conservamos solo la fila con menor section_id.
--
-- Usamos una tabla temporal para evitar problemas de performance
-- con subconsultas correlacionadas en BDs grandes.

CREATE TEMP TABLE _dups_to_delete AS
SELECT s2.section_id
FROM sections s1
JOIN sections s2
  ON s1.family_code = s2.family_code
 AND COALESCE(s1.subclass, '') = COALESCE(s2.subclass, '')
 AND s1.designation_modern = s2.designation_modern
 AND s1.source_catalog = s2.source_catalog
 AND s1.section_id < s2.section_id;

-- Reportar cuántos se van a eliminar (visible en logs si se ejecuta
-- con sqlite3 CLI)
SELECT COUNT(*) AS duplicates_to_delete FROM _dups_to_delete;

-- Eliminar (ON DELETE CASCADE propaga a tablas hijas)
DELETE FROM sections
WHERE section_id IN (SELECT section_id FROM _dups_to_delete);

-- Limpiar tabla temporal
DROP TABLE _dups_to_delete;

-- ------------------------------------------------------------
-- Step 2: Crear índice UNIQUE basado en expresión
-- ------------------------------------------------------------
-- Este índice trata subclass=NULL como '' (string vacío), por lo que
-- dos filas con subclass=NULL y mismo (family_code, designation_modern,
-- source_catalog) se consideran duplicadas y el INSERT falla.
--
-- La aplicación debe usar ON CONFLICT DO NOTHING o un pre-check SELECT
-- con COALESCE para detectar duplicados correctamente.

DROP INDEX IF EXISTS idx_sections_unique_designation;
CREATE UNIQUE INDEX idx_sections_unique_designation
ON sections (family_code, COALESCE(subclass, ''), designation_modern, source_catalog);

-- ------------------------------------------------------------
-- Step 3: Actualizar versión del esquema
-- ------------------------------------------------------------
UPDATE schema_meta SET value = '1.0.1' WHERE key = 'schema_version';

INSERT OR IGNORE INTO catalog_versions (version, installed_at, source_filename, checksum_sha256, profile_count, notes)
SELECT '1.0.1', datetime('now'), 'migration_003', '0000000000000000000000000000000000000000000000000000000000000000',
       (SELECT COUNT(*) FROM sections),
       'Migración 003: deduplicación + índice UNIQUE basado en expresión para tratar subclass=NULL correctamente';
