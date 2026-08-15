-- ============================================================
-- Migración 009: Eliminar perfiles T (ICHA) con datos corruptos
-- Versión: 1.0.7
-- ============================================================
-- Problema:
--   111 de 153 perfiles de la familia T (source_catalog='ICHA') tienen
--   tf = bf (o tf = d), un bug de mapeo del parser original que duplicó
--   una columna en vez de guardar el espesor de ala real.
--
--   A diferencia del bug W/SACK (migración 008), aquí NO es un simple
--   desplazamiento de columna reconstruible: se verificó algebraicamente
--   que area_mm2 y weight_kg_m almacenados tampoco son consistentes con
--   NINGÚN valor positivo de tf dado d/bf/tw (la ecuación de área de tee
--   A = bf·tf + (d-tf)·tw da tf negativo para las 111 filas revisadas).
--   Es decir, los datos de fuente ICHA para esta familia están corruptos
--   más allá de lo reconstruible sin la tabla ICHA original.
--
--   Los 42 perfiles T de fuente 'Bechtel' NO están afectados (tf real,
--   consistente con bf/tw) y permanecen en el catálogo.
--
-- Decisión (confirmada con el usuario): ocultar del catálogo estas 111
-- filas hasta contar con datos ICHA correctos, en vez de fabricar
-- dimensiones sin una fuente confiable (violaría el principio de que el
-- catálogo debe ser referencia real, no inventada).
--
-- ON DELETE CASCADE en las FK de section_dimensions/properties_*/
-- material_links y los triggers FTS (sections_ad) limpian automáticamente
-- las tablas dependientes y el índice de búsqueda.
-- ============================================================

BEGIN TRANSACTION;

DELETE FROM sections
WHERE section_id IN (
    SELECT s.section_id
    FROM sections s
    JOIN section_dimensions sd ON sd.section_id = s.section_id
    WHERE s.family_code = 'T'
      AND s.source_catalog = 'ICHA'
      AND (sd.tf = sd.bf OR sd.tf = sd.d)
);

INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('schema_version', '1.0.7');
INSERT OR REPLACE INTO schema_meta (key, value)
    VALUES ('migration_009_applied', datetime('now'));

COMMIT;
