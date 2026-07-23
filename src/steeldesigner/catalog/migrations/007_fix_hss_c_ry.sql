-- ============================================================
-- Migración 007: Corregir ry_mm para secciones HSS_C
-- Versión: 1.0.5
-- ============================================================
-- Problema:
--   La migración 006 usa COALESCE(ry_mm, rx_mm), por lo que no
--   sobreescribe ry cuando ya existe (valor inicial 1.0 del parser).
--   Para tubos circulares, ry = rx por simetría.
-- ============================================================

BEGIN TRANSACTION;

UPDATE section_properties_weak
SET
    ry_mm  = (SELECT sps.rx_mm  FROM section_properties_strong sps WHERE sps.section_id = section_properties_weak.section_id),
    Iy_mm4 = (SELECT sps.Ix_mm4 FROM section_properties_strong sps WHERE sps.section_id = section_properties_weak.section_id),
    Sy_mm3 = (SELECT sps.Sx_mm3 FROM section_properties_strong sps WHERE sps.section_id = section_properties_weak.section_id),
    Zy_mm3 = (SELECT sps.Zx_mm3 FROM section_properties_strong sps WHERE sps.section_id = section_properties_weak.section_id)
WHERE section_id IN (
    SELECT section_id FROM sections WHERE family_code = 'HSS_C'
);

INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('schema_version', '1.0.5');
INSERT OR REPLACE INTO schema_meta (key, value)
    VALUES ('migration_007_applied', datetime('now'));

COMMIT;
