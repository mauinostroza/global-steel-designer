-- ============================================================
-- Migración 006: Derivar módulos seccionales faltantes
-- Versión: 1.0.4
-- Fecha: 2026-07-03
-- ============================================================
-- Problema:
--   Después de la migración 005 (corrección de unidades), varias secciones
--   tienen Ix correcto pero Sx, Zx o rx nulos. Se pueden derivar
--   geométricamente para los casos más comunes.
--
-- Secciones afectadas:
--   HSS_C  — Sx/Zx/rx derivados en 005, J=2·Ix. Esta migración los pulsa
--             para secciones que quedaron incompletas.
--   CJ/CJE — Sx = Ix/(d/2) para secciones donde Sx IS NULL
--   Iy = Ix para perfiles simétricos cuadrados (HSS_C, CJ cuadrado)
-- ============================================================

BEGIN TRANSACTION;

-- ─────────────────────────────────────────────
-- 1. CJ cuadrados — Sy, Zy, ry (Iy=Ix para sección cuadrada)
-- ─────────────────────────────────────────────
-- Para CJ cuadrado (d=bf): Iy = Ix, Sy = Sx, Zy = Zx
UPDATE section_properties_weak
SET
    Iy_mm4 = COALESCE(Iy_mm4, (
        SELECT sp.Ix_mm4 FROM section_properties_strong sp
        WHERE sp.section_id = section_properties_weak.section_id
    )),
    Sy_mm3 = COALESCE(Sy_mm3, (
        SELECT sp.Sx_mm3 FROM section_properties_strong sp
        WHERE sp.section_id = section_properties_weak.section_id
    )),
    Zy_mm3 = COALESCE(Zy_mm3, (
        SELECT sp.Zx_mm3 FROM section_properties_strong sp
        WHERE sp.section_id = section_properties_weak.section_id
    )),
    ry_mm  = COALESCE(ry_mm, (
        SELECT sp.rx_mm FROM section_properties_strong sp
        WHERE sp.section_id = section_properties_weak.section_id
    ))
WHERE section_id IN (
    SELECT s.section_id
    FROM sections s
    JOIN section_dimensions sd ON s.section_id = sd.section_id
    WHERE s.family_code IN ('CJ', 'CJE', 'HSS_C')
      AND ABS(sd.d - sd.bf) < 1.0  -- sección cuadrada/circular: d ≈ bf
);

-- ─────────────────────────────────────────────
-- 2. CJ rectangulares — Sx y rx donde falta
-- ─────────────────────────────────────────────
UPDATE section_properties_strong
SET
    Sx_mm3 = CASE
        WHEN Sx_mm3 IS NULL AND Ix_mm4 IS NOT NULL AND sd.d > 0
        THEN Ix_mm4 / (sd.d / 2.0)
        ELSE Sx_mm3
    END,
    rx_mm = CASE
        WHEN rx_mm IS NULL AND Ix_mm4 IS NOT NULL AND sd.area_mm2 > 0
        THEN SQRT(Ix_mm4 / sd.area_mm2)
        ELSE rx_mm
    END
FROM section_dimensions sd
WHERE section_properties_strong.section_id = sd.section_id
  AND sd.section_id IN (
      SELECT section_id FROM sections WHERE family_code IN ('CJ', 'CJE')
  );

-- ─────────────────────────────────────────────
-- 3. Sy, ry para CJ rectangulares (Iy puede diferir de Ix)
-- ─────────────────────────────────────────────
UPDATE section_properties_weak
SET
    Sy_mm3 = CASE
        WHEN Sy_mm3 IS NULL AND Iy_mm4 IS NOT NULL AND sd.bf > 0
        THEN Iy_mm4 / (sd.bf / 2.0)
        ELSE Sy_mm3
    END,
    ry_mm = CASE
        WHEN ry_mm IS NULL AND Iy_mm4 IS NOT NULL AND sd.area_mm2 > 0
        THEN SQRT(Iy_mm4 / sd.area_mm2)
        ELSE ry_mm
    END
FROM section_dimensions sd
WHERE section_properties_weak.section_id = sd.section_id
  AND sd.section_id IN (
      SELECT section_id FROM sections WHERE family_code IN ('CJ', 'CJE')
  );

-- ─────────────────────────────────────────────
-- 4. Registrar versión
-- ─────────────────────────────────────────────
INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('schema_version', '1.0.4');
INSERT OR REPLACE INTO schema_meta (key, value)
    VALUES ('migration_006_applied', datetime('now'));

COMMIT;
