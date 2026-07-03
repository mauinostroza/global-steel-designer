-- ============================================================
-- Migración 005: Corregir errores de unidades en datos ingresados
-- Versión: 1.0.3
-- Fecha: 2026-07-03
-- ============================================================
-- Problema:
--   Varios parsers de ingesta guardaron valores en unidades fuente
--   (pulgadas, cm, cm²) en columnas que el esquema documenta como mm/mm²/mm⁴.
--   Normalizer en catalog/normalizer.py tiene todos los factores pero
--   los parsers nunca los llamaron para las familias afectadas.
--
-- Familias corregidas:
--   HSS_C  — d, bf en pulgadas; tf contiene Ix (bug de mapeo de columnas)
--   CJ/CJE — secciones pequeñas (d<50): d, bf en pulgadas
--   IP     — area_mm2 en cm² (fuente Bechtel)
--   W/SACK — d, bf en cm para secciones con d<50 almacenado
--   L_ICHA_LAM — area_mm2 contiene weight_kg_m (error de parser)
--
-- Nota de seguridad:
--   Todas las condiciones WHERE son conservadoras para no tocar secciones
--   que ya tienen unidades correctas (p.ej. CJ 400x100 tiene d=400 en mm).
-- ============================================================

BEGIN TRANSACTION;

-- ─────────────────────────────────────────────
-- 1. HSS_C — corregir d y bf: pulgadas → mm
-- ─────────────────────────────────────────────
-- Evidencia: "O 10x109.0" tiene d=10.0 (10 pulgadas = 254 mm). Todas
-- las 123 secciones HSS_C tienen d entre 2.77 y 12.0 → pulgadas crudas.
UPDATE section_dimensions
SET
    d  = d  * 25.4,
    bf = bf * 25.4,
    tf = NULL,    -- tf contenía el valor de Ix_mm4 (bug de mapeo del parser)
    tw = NULL     -- tubo circular: tw no aplica
WHERE section_id IN (
    SELECT section_id FROM sections WHERE family_code = 'HSS_C'
);

-- ─────────────────────────────────────────────
-- 2. HSS_C — derivar t_des desde área y d corregido
-- ─────────────────────────────────────────────
-- Para tubo circular: A = π/4 × (D² - (D-2t)²) → t = D/2 - sqrt(D²/4 - A/π)
UPDATE section_dimensions
SET t_des = (d / 2.0) - SQRT(MAX(0.0, (d * d / 4.0) - (area_mm2 / 3.14159265358979323846)))
WHERE section_id IN (
    SELECT section_id FROM sections WHERE family_code = 'HSS_C'
)
AND area_mm2 > 0 AND d > 0;

-- ─────────────────────────────────────────────
-- 3. HSS_C — derivar Sx, Zx y J usando Ix ya correcto
-- ─────────────────────────────────────────────
-- Los Ix_mm4 en section_properties_strong para HSS_C también son incorrectos
-- (el parser guardó basura allí). Recalculamos desde geometría usando t_des
-- ya derivado en el paso anterior:
-- Ix = π/64 × (D⁴ - (D-2t)⁴)
-- Para SQLite: π ≈ 3.14159265358979323846, d⁴ = d*d*d*d
UPDATE section_properties_strong
SET
    Ix_mm4 = (3.14159265358979323846 / 64.0) *
              (sd.d*sd.d*sd.d*sd.d - (sd.d - 2.0*sd.t_des)*(sd.d - 2.0*sd.t_des)*(sd.d - 2.0*sd.t_des)*(sd.d - 2.0*sd.t_des)),
    Sx_mm3 = ((3.14159265358979323846 / 64.0) *
               (sd.d*sd.d*sd.d*sd.d - (sd.d - 2.0*sd.t_des)*(sd.d - 2.0*sd.t_des)*(sd.d - 2.0*sd.t_des)*(sd.d - 2.0*sd.t_des)))
              / (sd.d / 2.0),
    Zx_mm3 = ((sd.d*sd.d*sd.d) - (sd.d - 2.0*sd.t_des)*(sd.d - 2.0*sd.t_des)*(sd.d - 2.0*sd.t_des)) / 6.0,
    rx_mm  = SQRT(MAX(1e-9,
              ((3.14159265358979323846 / 64.0) *
               (sd.d*sd.d*sd.d*sd.d - (sd.d - 2.0*sd.t_des)*(sd.d - 2.0*sd.t_des)*(sd.d - 2.0*sd.t_des)*(sd.d - 2.0*sd.t_des)))
              / MAX(sd.area_mm2, 1e-9)))
FROM section_dimensions sd
WHERE section_properties_strong.section_id = sd.section_id
  AND sd.section_id IN (
      SELECT section_id FROM sections WHERE family_code = 'HSS_C'
  )
  AND sd.t_des IS NOT NULL AND sd.t_des > 0 AND sd.d > 0;

-- Para J = 2·Ix (propiedad exacta del círculo)
UPDATE section_properties_torsion
SET J_mm4 = (
    SELECT 2.0 * sp.Ix_mm4
    FROM section_properties_strong sp
    WHERE sp.section_id = section_properties_torsion.section_id
      AND sp.Ix_mm4 IS NOT NULL
)
WHERE section_id IN (
    SELECT section_id FROM sections WHERE family_code = 'HSS_C'
);

-- ─────────────────────────────────────────────
-- 4. CJ/CJE pequeños — d, bf: pulgadas → mm
-- ─────────────────────────────────────────────
-- Criterio: d < 50 → definitivamente en pulgadas (máximo nominal 20"=508mm,
-- pero las secciones incorrectas tienen d=2,3,4,5,6,8,12,15,20 en pulgadas).
-- Las CJ grandes (CJ 400x100) tienen d=400 → no se tocan.
UPDATE section_dimensions
SET
    d  = d  * 25.4,
    bf = bf * 25.4
WHERE section_id IN (
    SELECT s.section_id
    FROM sections s
    JOIN section_dimensions sd ON s.section_id = sd.section_id
    WHERE s.family_code IN ('CJ', 'CJE') AND sd.d < 50
);

-- ─────────────────────────────────────────────
-- 5. IP (Bechtel) — area_mm2: cm² → mm²
-- ─────────────────────────────────────────────
-- Evidencia: IP 1000x350x246 tiene area=313 (= 313 cm² = 31,300 mm²).
-- Criterio: area < 2000 en las secciones IP (en mm² real serían >5000).
UPDATE section_dimensions
SET area_mm2 = area_mm2 * 100.0
WHERE section_id IN (
    SELECT section_id FROM sections WHERE family_code = 'IP'
)
AND area_mm2 < 2000;

-- Recalcular rx, ry con el área corregida
UPDATE section_properties_strong
SET rx_mm = CASE
    WHEN sd.area_mm2 > 0 AND Ix_mm4 IS NOT NULL
    THEN SQRT(Ix_mm4 / sd.area_mm2)
    ELSE rx_mm
END
FROM section_dimensions sd
WHERE section_properties_strong.section_id = sd.section_id
  AND sd.section_id IN (SELECT section_id FROM sections WHERE family_code = 'IP');

-- ─────────────────────────────────────────────
-- 6. W (fuente SACK) — d, bf: cm → mm
-- ─────────────────────────────────────────────
-- Evidencia: W 150x22.5 de SACK tiene d=15.0 (= 15 cm = 150 mm).
-- Criterio: source_catalog='SACK' AND d < 50 (en mm sería < 500 mm posible
-- pero los valores incorrectos tienen d=15-99 cm → mm crudos < 100 siempre).
UPDATE section_dimensions
SET
    d  = d  * 10.0,
    bf = bf * 10.0
WHERE section_id IN (
    SELECT s.section_id
    FROM sections s
    JOIN section_dimensions sd ON s.section_id = sd.section_id
    WHERE s.family_code = 'W' AND s.source_catalog = 'SACK' AND sd.d < 50
);

-- ─────────────────────────────────────────────
-- 7. L_ICHA_LAM — area_mm2 contiene weight_kg_m
-- ─────────────────────────────────────────────
-- Evidencia: los 8 ángulos laminados ICHA tienen area entre 4 y 35
-- (= weight_kg_m del perfil). Área real = weight / (ρ × 1m) = weight×1000/7.85.
-- Criterio: area_mm2 < 100 (área real de L20x20 ≈ 111 mm²).
UPDATE section_dimensions
SET area_mm2 = weight_kg_m * 1000.0 / 7.85
WHERE section_id IN (
    SELECT section_id FROM sections WHERE family_code = 'L_ICHA_LAM'
)
AND weight_kg_m > 0 AND area_mm2 < 100;

-- ─────────────────────────────────────────────
-- 8. Registrar versión del schema
-- ─────────────────────────────────────────────
INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('schema_version', '1.0.3');
INSERT OR REPLACE INTO schema_meta (key, value)
    VALUES ('migration_005_applied', datetime('now'));

COMMIT;
