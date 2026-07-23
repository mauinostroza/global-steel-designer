-- ============================================================
-- Migración 008: Corregir mapeo de columnas W/SACK
-- Versión: 1.0.6
-- ============================================================
-- Problema:
--   24 perfiles W (source_catalog='SACK') tienen area_mm2=0 y columnas
--   completamente permutadas por un bug del parser original:
--     stored d   = peso en lb/ft (valor descartable)
--     stored bf  = peso en kg/m
--     stored tf  = d real (mm)
--     stored tw  = bf real (mm)
--     tf y tw reales (espesores) nunca se almacenaron
--
--   Los 57 perfiles restantes ya tienen columnas correctas (area>0).
--
-- Fix:
--   Para cada designación afectada, se fijan d, bf, tf y tw con los
--   valores de la AISC Shapes Database 14ª ed. (secciones estándar
--   equivalentes métricas), y area_mm2 se deriva del peso: A = w*1000/7.85.
--
-- Nota: los perfiles W/SACK correctos NO se tocan (WHERE area_mm2=0).
-- ============================================================

BEGIN TRANSACTION;

-- ─────────────────────────────────────────────
-- W6 / W150 series (equivalentes AISC)
-- ─────────────────────────────────────────────

-- W 150x22.5 → W6x15: d=152mm, bf=152mm, tf=6.60mm, tw=5.84mm
UPDATE section_dimensions
SET d=152.0, bf=152.0, tf=6.60, tw=5.84,
    area_mm2=ROUND(weight_kg_m*1000.0/7.85, 1)
WHERE section_id=(SELECT section_id FROM sections WHERE designation_modern='W 150x22.5' AND source_catalog='SACK')
  AND area_mm2=0;

-- W 150x29.8 → W6x20: d=157mm, bf=153mm, tf=9.27mm, tw=6.60mm
UPDATE section_dimensions
SET d=157.0, bf=153.0, tf=9.27, tw=6.60,
    area_mm2=ROUND(weight_kg_m*1000.0/7.85, 1)
WHERE section_id=(SELECT section_id FROM sections WHERE designation_modern='W 150x29.8' AND source_catalog='SACK')
  AND area_mm2=0;

-- W 150x37.1 → W6x25: d=162mm, bf=154mm, tf=11.6mm, tw=8.13mm
UPDATE section_dimensions
SET d=162.0, bf=154.0, tf=11.6, tw=8.13,
    area_mm2=ROUND(weight_kg_m*1000.0/7.85, 1)
WHERE section_id=(SELECT section_id FROM sections WHERE designation_modern='W 150x37.1' AND source_catalog='SACK')
  AND area_mm2=0;

-- ─────────────────────────────────────────────
-- W8 / W200 series
-- ─────────────────────────────────────────────

-- W 200x35.9 → W8x24: d=201mm, bf=165mm, tf=10.2mm, tw=6.22mm
UPDATE section_dimensions
SET d=201.0, bf=165.0, tf=10.2, tw=6.22,
    area_mm2=ROUND(weight_kg_m*1000.0/7.85, 1)
WHERE section_id=(SELECT section_id FROM sections WHERE designation_modern='W 200x35.9' AND source_catalog='SACK')
  AND area_mm2=0;

-- W 200x46.1 → W8x31: d=203mm, bf=203mm, tf=11.0mm, tw=7.24mm
UPDATE section_dimensions
SET d=203.0, bf=203.0, tf=11.0, tw=7.24,
    area_mm2=ROUND(weight_kg_m*1000.0/7.85, 1)
WHERE section_id=(SELECT section_id FROM sections WHERE designation_modern='W 200x46.1' AND source_catalog='SACK')
  AND area_mm2=0;

-- W 200x52.0 → W8x35: d=206mm, bf=204mm, tf=12.6mm, tw=7.87mm
UPDATE section_dimensions
SET d=206.0, bf=204.0, tf=12.6, tw=7.87,
    area_mm2=ROUND(weight_kg_m*1000.0/7.85, 1)
WHERE section_id=(SELECT section_id FROM sections WHERE designation_modern='W 200x52.0' AND source_catalog='SACK')
  AND area_mm2=0;

-- W 200x59.0 → W8x40: d=210mm, bf=205mm, tf=14.2mm, tw=9.14mm
UPDATE section_dimensions
SET d=210.0, bf=205.0, tf=14.2, tw=9.14,
    area_mm2=ROUND(weight_kg_m*1000.0/7.85, 1)
WHERE section_id=(SELECT section_id FROM sections WHERE designation_modern='W 200x59.0' AND source_catalog='SACK')
  AND area_mm2=0;

-- W 200x71.0 → W8x48: d=216mm, bf=206mm, tf=17.4mm, tw=10.2mm
UPDATE section_dimensions
SET d=216.0, bf=206.0, tf=17.4, tw=10.2,
    area_mm2=ROUND(weight_kg_m*1000.0/7.85, 1)
WHERE section_id=(SELECT section_id FROM sections WHERE designation_modern='W 200x71.0' AND source_catalog='SACK')
  AND area_mm2=0;

-- W 200x86.0 → W8x58: d=222mm, bf=209mm, tf=20.6mm, tw=13.0mm
UPDATE section_dimensions
SET d=222.0, bf=209.0, tf=20.6, tw=13.0,
    area_mm2=ROUND(weight_kg_m*1000.0/7.85, 1)
WHERE section_id=(SELECT section_id FROM sections WHERE designation_modern='W 200x86.0' AND source_catalog='SACK')
  AND area_mm2=0;

-- ─────────────────────────────────────────────
-- W10 / W250 series
-- ─────────────────────────────────────────────

-- W 250x73.0 → W10x49: d=253mm, bf=254mm, tf=14.2mm, tw=8.64mm
UPDATE section_dimensions
SET d=253.0, bf=254.0, tf=14.2, tw=8.64,
    area_mm2=ROUND(weight_kg_m*1000.0/7.85, 1)
WHERE section_id=(SELECT section_id FROM sections WHERE designation_modern='W 250x73.0' AND source_catalog='SACK')
  AND area_mm2=0;

-- W 250x80.0 → W10x54: d=256mm, bf=255mm, tf=15.6mm, tw=9.40mm
UPDATE section_dimensions
SET d=256.0, bf=255.0, tf=15.6, tw=9.40,
    area_mm2=ROUND(weight_kg_m*1000.0/7.85, 1)
WHERE section_id=(SELECT section_id FROM sections WHERE designation_modern='W 250x80.0' AND source_catalog='SACK')
  AND area_mm2=0;

-- W 250x89.0 → W10x60: d=260mm, bf=256mm, tf=17.3mm, tw=10.7mm
UPDATE section_dimensions
SET d=260.0, bf=256.0, tf=17.3, tw=10.7,
    area_mm2=ROUND(weight_kg_m*1000.0/7.85, 1)
WHERE section_id=(SELECT section_id FROM sections WHERE designation_modern='W 250x89.0' AND source_catalog='SACK')
  AND area_mm2=0;

-- W 250x101.0 → W10x68: d=264mm, bf=257mm, tf=19.6mm, tw=11.9mm
UPDATE section_dimensions
SET d=264.0, bf=257.0, tf=19.6, tw=11.9,
    area_mm2=ROUND(weight_kg_m*1000.0/7.85, 1)
WHERE section_id=(SELECT section_id FROM sections WHERE designation_modern='W 250x101.0' AND source_catalog='SACK')
  AND area_mm2=0;

-- W 250x115.0 → W10x77: d=269mm, bf=259mm, tf=22.1mm, tw=13.5mm
UPDATE section_dimensions
SET d=269.0, bf=259.0, tf=22.1, tw=13.5,
    area_mm2=ROUND(weight_kg_m*1000.0/7.85, 1)
WHERE section_id=(SELECT section_id FROM sections WHERE designation_modern='W 250x115.0' AND source_catalog='SACK')
  AND area_mm2=0;

-- ─────────────────────────────────────────────
-- W12 / W310 series
-- ─────────────────────────────────────────────

-- W 310x97.0 → W12x65: d=308mm, bf=305mm, tf=15.4mm, tw=9.91mm
UPDATE section_dimensions
SET d=308.0, bf=305.0, tf=15.4, tw=9.91,
    area_mm2=ROUND(weight_kg_m*1000.0/7.85, 1)
WHERE section_id=(SELECT section_id FROM sections WHERE designation_modern='W 310x97.0' AND source_catalog='SACK')
  AND area_mm2=0;

-- W 310x107.0 → W12x72: d=311mm, bf=306mm, tf=17.0mm, tw=10.9mm
UPDATE section_dimensions
SET d=311.0, bf=306.0, tf=17.0, tw=10.9,
    area_mm2=ROUND(weight_kg_m*1000.0/7.85, 1)
WHERE section_id=(SELECT section_id FROM sections WHERE designation_modern='W 310x107.0' AND source_catalog='SACK')
  AND area_mm2=0;

-- W 310x110.0 (≈W12x74, non-standard): d=308mm, bf=310mm, tf=17.0mm, tw=10.9mm
-- (dimensiones interpoladas; similar a W12x72 con bf ligeramente mayor)
UPDATE section_dimensions
SET d=308.0, bf=310.0, tf=17.0, tw=10.9,
    area_mm2=ROUND(weight_kg_m*1000.0/7.85, 1)
WHERE section_id=(SELECT section_id FROM sections WHERE designation_modern='W 310x110.0' AND source_catalog='SACK')
  AND area_mm2=0;

-- W 310x117.0 → W12x79: d=314mm, bf=307mm, tf=18.7mm, tw=11.9mm
UPDATE section_dimensions
SET d=314.0, bf=307.0, tf=18.7, tw=11.9,
    area_mm2=ROUND(weight_kg_m*1000.0/7.85, 1)
WHERE section_id=(SELECT section_id FROM sections WHERE designation_modern='W 310x117.0' AND source_catalog='SACK')
  AND area_mm2=0;

-- ─────────────────────────────────────────────
-- W14 / W360 series
-- ─────────────────────────────────────────────

-- W 360x91.0 → W14x61: d=353mm, bf=254mm, tf=16.4mm, tw=9.53mm
UPDATE section_dimensions
SET d=353.0, bf=254.0, tf=16.4, tw=9.53,
    area_mm2=ROUND(weight_kg_m*1000.0/7.85, 1)
WHERE section_id=(SELECT section_id FROM sections WHERE designation_modern='W 360x91.0' AND source_catalog='SACK')
  AND area_mm2=0;

-- W 360x101.0 → W14x68: d=357mm, bf=255mm, tf=17.9mm, tw=10.5mm
UPDATE section_dimensions
SET d=357.0, bf=255.0, tf=17.9, tw=10.5,
    area_mm2=ROUND(weight_kg_m*1000.0/7.85, 1)
WHERE section_id=(SELECT section_id FROM sections WHERE designation_modern='W 360x101.0' AND source_catalog='SACK')
  AND area_mm2=0;

-- W 360x110.0 → W14x74: d=360mm, bf=256mm, tf=18.8mm, tw=11.2mm
UPDATE section_dimensions
SET d=360.0, bf=256.0, tf=18.8, tw=11.2,
    area_mm2=ROUND(weight_kg_m*1000.0/7.85, 1)
WHERE section_id=(SELECT section_id FROM sections WHERE designation_modern='W 360x110.0' AND source_catalog='SACK')
  AND area_mm2=0;

-- W 360x122.0 → W14x82: d=363mm, bf=257mm, tf=21.7mm, tw=13.0mm
UPDATE section_dimensions
SET d=363.0, bf=257.0, tf=21.7, tw=13.0,
    area_mm2=ROUND(weight_kg_m*1000.0/7.85, 1)
WHERE section_id=(SELECT section_id FROM sections WHERE designation_modern='W 360x122.0' AND source_catalog='SACK')
  AND area_mm2=0;

-- ─────────────────────────────────────────────
-- W16 / W410 series
-- ─────────────────────────────────────────────

-- W 410x85.0 → W16x57: d=417mm, bf=181mm, tf=18.2mm, tw=10.9mm
-- Nota: para esta sección el mapeo es distinto (d=bf_stored, bf=tf_stored)
-- tw_stored ya contiene tw real (10.9mm). Solo se corrige d, bf y tf.
UPDATE section_dimensions
SET d=417.0, bf=181.0, tf=18.2, tw=10.9,
    area_mm2=ROUND(weight_kg_m*1000.0/7.85, 1)
WHERE section_id=(SELECT section_id FROM sections WHERE designation_modern='W 410x85.0' AND source_catalog='SACK')
  AND area_mm2=0;

-- ─────────────────────────────────────────────
-- Registrar versión
-- ─────────────────────────────────────────────
INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('schema_version', '1.0.6');
INSERT OR REPLACE INTO schema_meta (key, value)
    VALUES ('migration_008_applied', datetime('now'));

COMMIT;
