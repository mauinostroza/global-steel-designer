-- ============================================================
-- Migración 001: Esquema inicial del catálogo de secciones
-- Versión: 1.0.0
-- Fecha: 2026-06-21
-- ============================================================
-- Crea 12 tablas normalizadas + 1 tabla virtual FTS5 + triggers
-- para mantener el índice de texto completo sincronizado.
-- ============================================================

-- Asegurar claves foráneas y modo WAL (la app debe aplicar PRAGMA
-- journal_mode=WAL por separado al abrir la conexión).
PRAGMA foreign_keys = ON;

-- ------------------------------------------------------------
-- 1. families: familias de perfiles (IN, HN, W, IPE, HEA, ...)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS families (
    family_code           TEXT PRIMARY KEY,
    name_es               TEXT NOT NULL,
    name_en               TEXT,
    manufacturing_process TEXT NOT NULL
                          CHECK (manufacturing_process IN
                                 ('hot_rolled','cold_formed','welded','back_to_back','cut_from_parent')),
    source_standard       TEXT NOT NULL,
    source_edition        TEXT,
    is_chilean            INTEGER NOT NULL DEFAULT 0,
    has_subclasses        INTEGER NOT NULL DEFAULT 0,
    drawing_template      TEXT NOT NULL
                          CHECK (drawing_template IN
                                 ('I_welded','I_rolled','channel_rolled','channel_cf',
                                  'angle','hss_rect','hss_circ','tee'))
);

-- ------------------------------------------------------------
-- 2. sections: una fila por perfil único
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sections (
    section_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    family_code           TEXT NOT NULL REFERENCES families(family_code),
    subclass              TEXT,
    designation_modern    TEXT NOT NULL,
    designation_legacy    TEXT,
    designation_aisc      TEXT,
    designation_en        TEXT,
    source_catalog        TEXT NOT NULL
                          CHECK (source_catalog IN
                                 ('ICHA','Bechtel','SACK','CINTAC','AISC','EN','USER')),
    source_edition        TEXT,
    available_sack        INTEGER NOT NULL DEFAULT 0,
    available_cintac      INTEGER NOT NULL DEFAULT 0,
    is_standard           INTEGER NOT NULL DEFAULT 1,
    is_custom             INTEGER NOT NULL DEFAULT 0,
    notes                 TEXT,
    created_at            TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at            TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (family_code, subclass, designation_modern, source_catalog)
);

CREATE INDEX IF NOT EXISTS idx_sections_family         ON sections(family_code);
CREATE INDEX IF NOT EXISTS idx_sections_subclass       ON sections(subclass);
CREATE INDEX IF NOT EXISTS idx_sections_designation_m  ON sections(designation_modern);
CREATE INDEX IF NOT EXISTS idx_sections_designation_l  ON sections(designation_legacy);
CREATE INDEX IF NOT EXISTS idx_sections_designation_a  ON sections(designation_aisc);
CREATE INDEX IF NOT EXISTS idx_sections_designation_e  ON sections(designation_en);
CREATE INDEX IF NOT EXISTS idx_sections_source         ON sections(source_catalog);
CREATE INDEX IF NOT EXISTS idx_sections_custom         ON sections(is_custom) WHERE is_custom = 1;

-- ------------------------------------------------------------
-- 3. section_dimensions: geometría del perfil (todas en mm)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS section_dimensions (
    section_id            INTEGER PRIMARY KEY REFERENCES sections(section_id) ON DELETE CASCADE,
    d                     REAL,
    bf                    REAL,
    tf                    REAL,
    tw                    REAL,
    h                     REAL,
    T_clear               REAL,
    k                     REAL,
    k_design              REAL,
    k_detail              REAL,
    r                     REAL,           -- radio de empalme (laminados)
    R_ext                 REAL,           -- radio exterior (plegados CF)
    R_int                 REAL,           -- radio interior (plegados CF)
    B                     REAL,
    C_dim                 REAL,
    t_nom                 REAL,
    t_des                 REAL,
    area_mm2              REAL NOT NULL,
    perimeter_mm          REAL,
    weight_kg_m           REAL NOT NULL,
    is_hollow             INTEGER NOT NULL DEFAULT 0,
    is_sym_x              INTEGER NOT NULL DEFAULT 1,
    is_sym_y              INTEGER NOT NULL DEFAULT 1,
    centroid_x_mm         REAL DEFAULT 0,
    centroid_y_mm         REAL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_dims_d        ON section_dimensions(d);
CREATE INDEX IF NOT EXISTS idx_dims_bf       ON section_dimensions(bf);
CREATE INDEX IF NOT EXISTS idx_dims_weight   ON section_dimensions(weight_kg_m);
CREATE INDEX IF NOT EXISTS idx_dims_area     ON section_dimensions(area_mm2);

-- ------------------------------------------------------------
-- 4. section_properties_strong: eje X-X (eje fuerte)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS section_properties_strong (
    section_id            INTEGER PRIMARY KEY REFERENCES sections(section_id) ON DELETE CASCADE,
    Ix_mm4                REAL,
    Sx_mm3                REAL,
    Zx_mm3                REAL,
    rx_mm                 REAL,
    xp_mm                 REAL
);

-- ------------------------------------------------------------
-- 5. section_properties_weak: eje Y-Y (eje débil)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS section_properties_weak (
    section_id            INTEGER PRIMARY KEY REFERENCES sections(section_id) ON DELETE CASCADE,
    Iy_mm4                REAL,
    Sy_mm3                REAL,
    Zy_mm3                REAL,
    ry_mm                 REAL,
    yp_mm                 REAL
);

-- ------------------------------------------------------------
-- 6. section_properties_torsion: torsión y alabeo (sparse)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS section_properties_torsion (
    section_id            INTEGER PRIMARY KEY REFERENCES sections(section_id) ON DELETE CASCADE,
    J_mm4                 REAL,
    Cw_mm6                REAL,
    Wno                   REAL,
    Sw                    REAL,
    Qf                    REAL,
    Qw                    REAL,
    H_const               REAL,
    ro_mm                 REAL,
    xo_mm                 REAL,
    io_mm                 REAL,
    beta                  REAL,
    j                     REAL
);

-- ------------------------------------------------------------
-- 7. section_properties_local_buckling: esbeltez y Qs/Qa
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS section_properties_local_buckling (
    section_id            INTEGER PRIMARY KEY REFERENCES sections(section_id) ON DELETE CASCADE,
    bf_2tf                REAL,
    h_tw                  REAL,
    b_t                   REAL,
    D_t                   REAL,
    Qs                    REAL,
    Qa                    REAL,
    ia                    REAL,
    it                    REAL,
    X1                    REAL,
    Fy_default_MPa        REAL
);

-- ------------------------------------------------------------
-- 8. composite_separations: propiedades para series back-to-back
--    (IC, ICA, OC, OCA, 2L) según separación entre componentes
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS composite_separations (
    section_id            INTEGER NOT NULL REFERENCES sections(section_id) ON DELETE CASCADE,
    separation_mm         INTEGER NOT NULL,
    Iy_mm4                REAL,
    iy_mm                 REAL,
    J_mm4                 REAL,
    Cw_mm6                REAL,
    PRIMARY KEY (section_id, separation_mm)
);

-- ------------------------------------------------------------
-- 9. section_load_capacity: tablas de carga Cintac
--    (sólo para C, CA, IC, OC, ICA, OCA de Cintac 2008)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS section_load_capacity (
    section_id            INTEGER NOT NULL REFERENCES sections(section_id) ON DELETE CASCADE,
    span_m                REAL NOT NULL,
    unbraced_length_m     REAL,
    P10_kN                REAL,
    Ph_kN                 REAL,
    R10_kN                REAL,
    Rh_kN                 REAL,
    V_kN                  REAL,
    MA_kNm                REAL,
    Mc_kNm                REAL,
    Mmax_kNm              REAL,
    My_kNm                REAL,
    LU_m                  REAL,
    L200_m                REAL,
    PRIMARY KEY (section_id, span_m, unbraced_length_m)
);

CREATE INDEX IF NOT EXISTS idx_load_section  ON section_load_capacity(section_id);

-- ------------------------------------------------------------
-- 10. materials: grados de acero
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS materials (
    material_code         TEXT PRIMARY KEY,
    standard              TEXT NOT NULL,
    grade                 TEXT,
    Fy_MPa                REAL NOT NULL,
    Fu_MPa                REAL,
    E_MPa                 REAL DEFAULT 200000,
    nu                    REAL DEFAULT 0.30,
    applicable_families   TEXT
);

-- ------------------------------------------------------------
-- 11. section_material_links: perfil ↔ material (muchos a muchos)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS section_material_links (
    section_id            INTEGER NOT NULL REFERENCES sections(section_id) ON DELETE CASCADE,
    material_code         TEXT NOT NULL REFERENCES materials(material_code),
    is_default            INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (section_id, material_code)
);

CREATE INDEX IF NOT EXISTS idx_sml_section  ON section_material_links(section_id);
CREATE INDEX IF NOT EXISTS idx_sml_material ON section_material_links(material_code);
CREATE INDEX IF NOT EXISTS idx_sml_default  ON section_material_links(material_code) WHERE is_default = 1;

-- ------------------------------------------------------------
-- 12. catalog_versions: historial de versiones instaladas
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS catalog_versions (
    version               TEXT PRIMARY KEY,
    installed_at          TEXT NOT NULL,
    source_filename       TEXT,
    checksum_sha256       TEXT NOT NULL,
    profile_count         INTEGER,
    notes                 TEXT
);

-- ------------------------------------------------------------
-- 13. schema_meta: metadatos internos del esquema
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS schema_meta (
    key                   TEXT PRIMARY KEY,
    value                 TEXT NOT NULL
);

INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('schema_version', '1.0.0');
INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('created_at', datetime('now'));

-- ------------------------------------------------------------
-- 14. FTS5: índice de texto completo para búsqueda fuzzy y full-text
-- ------------------------------------------------------------
CREATE VIRTUAL TABLE IF NOT EXISTS sections_fts USING fts5(
    section_id UNINDEXED,
    designation_modern,
    designation_legacy,
    designation_aisc,
    designation_en,
    family_code,
    content='sections',
    content_rowid='section_id',
    tokenize='unicode61 remove_diacritics 2'
);

-- Triggers para mantener el FTS sincronizado
CREATE TRIGGER IF NOT EXISTS sections_ai AFTER INSERT ON sections BEGIN
    INSERT INTO sections_fts(rowid, designation_modern, designation_legacy,
                             designation_aisc, designation_en, family_code)
    VALUES (new.section_id, new.designation_modern, new.designation_legacy,
            new.designation_aisc, new.designation_en, new.family_code);
END;

CREATE TRIGGER IF NOT EXISTS sections_ad AFTER DELETE ON sections BEGIN
    DELETE FROM sections_fts WHERE rowid = old.section_id;
END;

CREATE TRIGGER IF NOT EXISTS sections_au AFTER UPDATE ON sections BEGIN
    DELETE FROM sections_fts WHERE rowid = old.section_id;
    INSERT INTO sections_fts(rowid, designation_modern, designation_legacy,
                             designation_aisc, designation_en, family_code)
    VALUES (new.section_id, new.designation_modern, new.designation_legacy,
            new.designation_aisc, new.designation_en, new.family_code);
END;

-- ------------------------------------------------------------
-- 15. Vista: v_section_equivalents (matching dimensional ±1 mm)
--     Materializada como tabla cache para evitar CROSS JOIN en cada consulta
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS equivalents_cache (
    base_id               INTEGER NOT NULL,
    equivalent_id         INTEGER NOT NULL,
    d_diff_mm             REAL,
    bf_diff_mm            REAL,
    tf_diff_mm            REAL,
    tw_diff_mm            REAL,
    PRIMARY KEY (base_id, equivalent_id),
    FOREIGN KEY (base_id) REFERENCES sections(section_id) ON DELETE CASCADE,
    FOREIGN KEY (equivalent_id) REFERENCES sections(section_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_eq_base       ON equivalents_cache(base_id);
CREATE INDEX IF NOT EXISTS idx_eq_equivalent ON equivalents_cache(equivalent_id);
