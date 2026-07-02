-- ============================================================
-- Seed: familias y materiales por defecto (v1.0.0)
-- ============================================================
-- Puebla las tablas families y materials con los registros base
-- que toda instalación del catálogo debe tener desde el arranque.
-- Los perfiles se ingestan en la Fase 2 (parsers ICHA, Bechtel,
-- SACK, Cintac, AISC, EN).
-- ============================================================

-- ------------------------------------------------------------
-- Familias (35 familias: 17 chilenas + 9 AISC + 9 EN)
-- ------------------------------------------------------------
INSERT OR IGNORE INTO families (family_code, name_es, name_en, manufacturing_process, source_standard, source_edition, is_chilean, has_subclasses, drawing_template) VALUES
    -- Familia unificada H soldados ICHA 2008 (IN, HN, IP, IE se unifican bajo H)
    ('H',  'H soldado (IN/HN/IP/IE unificado ICHA 2008)', 'Welded H-section (ICHA 2008 unified)', 'welded', 'ICHA', '2008', 1, 1, 'I_welded'),

    -- Familias chilenas (soldadas — preservadas en Bechtel 1976/2018)
    ('IN',  'I Normalizado soldado',           'Welded I-section normalized',     'welded',         'ICHA', '1976/2018', 1, 1, 'I_welded'),
    ('HN',  'H Normalizado soldado',           'Welded H-section normalized',     'welded',         'ICHA', '1976/2018', 1, 1, 'I_welded'),
    ('IP',  'I Pesado soldado',                'Welded I-section heavy',          'welded',         'ICHA', '1976/2018', 1, 1, 'I_welded'),
    ('IE',  'I Especial soldado',              'Welded I-section special',        'welded',         'ICHA', '1976/2018', 1, 1, 'I_welded'),
    ('PH',  'Pilotes H soldados',              'Welded H-piles',                  'welded',         'ICHA', '2008', 1, 0, 'I_welded'),
    ('HR',  'H Renormalizado (AISC W reexpr.)','Renormalized H (AISC W re-expressed)', 'hot_rolled', 'ICHA', '2008', 1, 0, 'I_rolled'),
    ('T',   'T soldado',                       'Welded tee',                      'welded',         'ICHA', '2008', 1, 0, 'tee'),

    -- Familias chilenas (conformadas en frío)
    ('C',   'Canal conformado en frio',        'Cold-formed channel',             'cold_formed',    'ICHA', '2008', 1, 0, 'channel_cf'),
    ('CA',  'Costanera atiesada CF',           'Cold-formed lipped channel',      'cold_formed',    'ICHA', '2008', 1, 0, 'channel_cf'),
    ('CJ',  'Cajon plegado',                   'Cold-formed box',                 'cold_formed',    'ICHA', '2008', 1, 0, 'hss_rect'),
    ('CJE', 'Cajon plegado especial',          'Cold-formed box special',         'cold_formed',    'ICHA', '1976/2018', 1, 0, 'hss_rect'),

    -- Familias chilenas (back-to-back, Cintac 2008)
    ('IC',  'Canal doble espalda',             'Back-to-back channels',           'back_to_back',   'CINTAC', '2008', 1, 0, 'channel_cf'),
    ('ICA', 'Costanera doble espalda',         'Back-to-back lipped channels',    'back_to_back',   'CINTAC', '2008', 1, 0, 'channel_cf'),
    ('OC',  'Cajon doble (2C)',                'Box from 2 channels',             'back_to_back',   'CINTAC', '2008', 1, 0, 'hss_rect'),
    ('OCA', 'Cajon doble (2CA)',               'Box from 2 lipped channels',      'back_to_back',   'CINTAC', '2008', 1, 0, 'hss_rect'),

    -- Familias chilenas (laminadas y plegadas en caliente)
    ('L_ICHA_PLEG', 'L plegada ICHA',          'ICHA cold-formed angle',          'cold_formed',    'ICHA', '2008', 1, 0, 'angle'),
    ('L_ICHA_LAM',  'L laminada ICHA',         'ICHA hot-rolled angle',           'hot_rolled',     'ICHA', '2008', 1, 0, 'angle'),

    -- Familias AISC
    ('W',   'Wide-flange AISC',                'AISC wide-flange',                'hot_rolled',     'AISC 360', 'v16', 0, 0, 'I_rolled'),
    ('HP',  'Pilotes AISC',                    'AISC H-piles',                    'hot_rolled',     'AISC 360', 'v16', 0, 0, 'I_rolled'),
    ('WT',  'Tee cortada de W',                'Tee cut from W',                  'cut_from_parent','AISC 360', 'v16', 0, 0, 'tee'),
    ('MT',  'Tee cortada de M',                'Tee cut from M',                  'cut_from_parent','AISC 360', 'v16', 0, 0, 'tee'),
    ('ST',  'Tee cortada de S',                'Tee cut from S',                  'cut_from_parent','AISC 360', 'v16', 0, 0, 'tee'),
    ('MC',  'Canal AISC',                      'AISC channel',                    'hot_rolled',     'AISC 360', 'v16', 0, 0, 'channel_rolled'),
    ('L_AISC', 'Angulo AISC',                  'AISC angle',                      'hot_rolled',     'AISC 360', 'v16', 0, 0, 'angle'),
    ('HSS_R','HSS rectangular AISC',           'AISC rectangular HSS',            'cold_formed',    'AISC 360', 'v16', 0, 0, 'hss_rect'),
    ('HSS_C','HSS circular AISC',              'AISC circular HSS',               'cold_formed',    'AISC 360', 'v16', 0, 0, 'hss_circ'),

    -- Familias europeas
    ('IPE', 'IPE europea',                     'European IPE',                    'hot_rolled',     'EN 10365', '2017', 0, 1, 'I_rolled'),
    ('IPN', 'IPN europea',                     'European IPN',                    'hot_rolled',     'EN 10365', '2017', 0, 0, 'I_rolled'),
    ('HEA', 'HEA europea',                     'European HEA',                    'hot_rolled',     'EN 10365', '2017', 0, 1, 'I_rolled'),
    ('HEB', 'HEB europea',                     'European HEB',                    'hot_rolled',     'EN 10365', '2017', 0, 0, 'I_rolled'),
    ('HEM', 'HEM europea',                     'European HEM',                    'hot_rolled',     'EN 10365', '2017', 0, 0, 'I_rolled'),
    ('HL',  'HL europea pesada',               'European HL',                     'hot_rolled',     'EN 10365', '2017', 0, 1, 'I_rolled'),
    ('HD',  'HD europea',                      'European HD',                     'hot_rolled',     'EN 10365', '2017', 0, 0, 'I_rolled'),
    ('UPN', 'UPN europea',                     'European UPN',                    'hot_rolled',     'EN 10365', '2017', 0, 0, 'channel_rolled'),
    ('UPE', 'UPE europea',                     'European UPE',                    'hot_rolled',     'EN 10365', '2017', 0, 0, 'channel_rolled');

-- ------------------------------------------------------------
-- Materiales (grados de acero)
-- ------------------------------------------------------------
INSERT OR IGNORE INTO materials (material_code, standard, grade, Fy_MPa, Fu_MPa, E_MPa, nu, applicable_families) VALUES
    ('A42-27ES',  'NCh 203',     'A42-27ES',       265.0, 412.0, 200000, 0.30,
     'C,CA,IC,ICA,OC,OCA,CJ,CJE,L_ICHA_PLEG'),
    ('A37-24ES',  'NCh 203',     'A37-24ES',       235.0, 370.0, 200000, 0.30,
     'C,CA,CJ,L_ICHA_PLEG'),
    ('A36',       'ASTM A36',    'A36',            250.0, 400.0, 200000, 0.30,
     'UPN,UPE,IPN,IPE,HEA,HEB,HEM,HL,HD,L_AISC,MC'),
    ('A572Gr50',  'ASTM A572',   'Gr 50',          345.0, 450.0, 200000, 0.30,
     'W,HP,WT,MT,ST,IN,HN,IP,IE,PH,HR,T,HSS_R,HSS_C'),
    ('A53',       'ASTM A53',    'A53',            240.0, 330.0, 200000, 0.30,
     'HSS_C'),
    ('A500GrB',   'ASTM A500',   'Gr B',           315.0, 400.0, 200000, 0.30,
     'HSS_R,HSS_C'),
    ('A500GrC',   'ASTM A500',   'Gr C',           345.0, 427.0, 200000, 0.30,
     'HSS_R,HSS_C');

-- ------------------------------------------------------------
-- Registrar versión inicial en catalog_versions
-- ------------------------------------------------------------
INSERT OR IGNORE INTO catalog_versions (version, installed_at, source_filename, checksum_sha256, profile_count, notes)
VALUES ('1.0.0', datetime('now'), 'seed', '0000000000000000000000000000000000000000000000000000000000000000',
        0, 'Esquema inicial v1.0.0 — familias y materiales cargados, sin perfiles todavía');
