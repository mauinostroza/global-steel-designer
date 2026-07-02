-- ============================================================
-- Migración 004: Fusionar duplicados cross-fuente
-- Versión: 1.0.2
-- Fecha: 2026-07-02
-- ============================================================
-- Problema:
--   El constraint UNIQUE(family_code, subclass, designation_modern,
--   source_catalog) evita duplicados exactos DENTRO de una misma fuente,
--   pero permite intencionalmente que el mismo perfil físico entre desde
--   varias fuentes (ej. EN 10365 y el catálogo comercial SACK). Cuando
--   dimensionalmente son el mismo perfil, verlo dos veces en el catálogo
--   se percibe como un duplicado real. Verificado en la BD: 44 grupos,
--   todos pares EN/SACK en las familias HEA, HEB e IPE.
--
-- Solución:
--   1. Para cada grupo de filas con igual (family_code, subclass tratando
--      NULL como '', designation_modern), elegir una fila "ganadora" por
--      prioridad de fuente: ICHA > Bechtel > EN > AISC > SACK > CINTAC >
--      USER (normas técnicas primero, catálogos comerciales redundantes
--      después; empate -> menor section_id).
--   2. Fusionar hacia la ganadora los flags de disponibilidad comercial
--      (available_sack, available_cintac = MAX(...) del grupo), para no
--      perder la señal de "disponible en SACK/Cintac" al descartar esas
--      filas.
--   3. Dejar traza en `notes` de qué fuente se fusionó.
--   4. Eliminar las filas perdedoras (ON DELETE CASCADE limpia las tablas
--      hijas: section_dimensions, section_properties_*, section_material_
--      links, y los triggers mantienen sincronizado sections_fts).
--
-- Nota: esta misma lógica se reimplementa en Python como
-- `Repository.dedupe_cross_source()` para poder invocarla también después
-- de cada regeneración completa del catálogo (`scripts/ingest_all.py`),
-- ya que esta migración solo corrige bases de datos ya existentes.
-- ============================================================

PRAGMA foreign_keys = ON;

-- ------------------------------------------------------------
-- Step 1: Rankear cada fila dentro de su grupo de duplicados por
-- prioridad de fuente (rn=1 es la ganadora que se conserva).
-- ------------------------------------------------------------
CREATE TEMP TABLE _dup_ranked AS
SELECT
    section_id,
    family_code,
    COALESCE(subclass, '') AS subclass_key,
    designation_modern,
    available_sack,
    available_cintac,
    ROW_NUMBER() OVER (
        PARTITION BY family_code, COALESCE(subclass, ''), designation_modern
        ORDER BY
            CASE source_catalog
                WHEN 'ICHA'    THEN 0
                WHEN 'Bechtel' THEN 1
                WHEN 'EN'      THEN 2
                WHEN 'AISC'    THEN 3
                WHEN 'SACK'    THEN 4
                WHEN 'CINTAC'  THEN 5
                ELSE 6
            END,
            section_id
    ) AS rn,
    COUNT(*) OVER (
        PARTITION BY family_code, COALESCE(subclass, ''), designation_modern
    ) AS group_size
FROM sections;

-- ------------------------------------------------------------
-- Step 2: Un renglón por grupo duplicado con el section_id ganador
-- y los flags de disponibilidad fusionados.
-- ------------------------------------------------------------
CREATE TEMP TABLE _dup_groups AS
SELECT
    family_code,
    subclass_key,
    designation_modern,
    MAX(CASE WHEN rn = 1 THEN section_id END) AS keeper_id,
    MAX(available_sack)   AS merged_available_sack,
    MAX(available_cintac) AS merged_available_cintac
FROM _dup_ranked
WHERE group_size > 1
GROUP BY family_code, subclass_key, designation_modern;

DROP TABLE _dup_ranked;

-- ------------------------------------------------------------
-- Step 3: Fusionar disponibilidad comercial y anotar traza en la ganadora
-- ------------------------------------------------------------
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
        'Fusionado (migración 004) con perfil(es) duplicado(s) de otra fuente: ' ||
        (SELECT GROUP_CONCAT(s2.source_catalog || ' section_id=' || s2.section_id, ', ')
         FROM sections s2
         JOIN _dup_groups g2
           ON g2.family_code = s2.family_code
          AND g2.subclass_key = COALESCE(s2.subclass, '')
          AND g2.designation_modern = s2.designation_modern
         WHERE g2.keeper_id = sections.section_id
           AND s2.section_id != sections.section_id)
WHERE section_id IN (SELECT keeper_id FROM _dup_groups);

-- ------------------------------------------------------------
-- Step 4: Eliminar las filas perdedoras (cascada a tablas hijas + FTS5)
-- ------------------------------------------------------------
DELETE FROM sections
WHERE section_id IN (
    SELECT s.section_id
    FROM sections s
    JOIN _dup_groups g
      ON g.family_code = s.family_code
     AND g.subclass_key = COALESCE(s.subclass, '')
     AND g.designation_modern = s.designation_modern
    WHERE s.section_id != g.keeper_id
);

DROP TABLE _dup_groups;

-- ------------------------------------------------------------
-- Step 5: Actualizar versión del esquema
-- ------------------------------------------------------------
UPDATE schema_meta SET value = '1.0.2' WHERE key = 'schema_version';

INSERT OR IGNORE INTO catalog_versions (version, installed_at, source_filename, checksum_sha256, profile_count, notes)
SELECT '1.0.2', datetime('now'), 'migration_004', '0000000000000000000000000000000000000000000000000000000000000000',
       (SELECT COUNT(*) FROM sections),
       'Migración 004: fusión de duplicados cross-fuente (mismo perfil físico en varios catálogos de origen)';
