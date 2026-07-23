"""
Cálculo de propiedades seccionales desde geometría bruta.

El catálogo solo provee dimensiones (d, bf, tf, tw, t_des, area_mm2).
Este módulo deriva Ix, Sx, Zx, rx, Iy, Sy, Zy, ry, J, Cw, rts, ro, etc.
en tiempo de ejecución, sin depender de los valores almacenados en la BD.

Referencia: AISC Steel Construction Manual 16ª ed., Apéndice B y Commentary.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class SectionProps:
    """Propiedades calculadas. Unidades: mm, mm², mm⁴, mm³, mm⁶."""
    area_mm2: float = 0.0
    Ix_mm4:   float = 0.0
    Sx_mm3:   float = 0.0
    Zx_mm3:   float = 0.0
    rx_mm:    float = 0.0
    Iy_mm4:   float = 0.0
    Sy_mm3:   float = 0.0
    Zy_mm3:   float = 0.0
    ry_mm:    float = 0.0
    J_mm4:    float = 0.0
    Cw_mm6:   float = 0.0
    rts_mm:   float = 0.0
    ro_mm:    float = 0.0
    h_tw:     float = 0.0
    bf_2tf:   float = 0.0
    D_t:      float = 0.0
    xo_mm:    float = 0.0   # distancia al centro de corte (canal)


def _pos(v: float, minimum: float = 1e-9) -> float:
    """Clamp a mínimo positivo para evitar división por cero."""
    return max(v, minimum)


# ─────────────────────────────────────────────────────────────────────────────
# I / H (doble T): W, HP, IPE, HEA, HEB, HEM, IP, IN, HR, HL, HD …
# ─────────────────────────────────────────────────────────────────────────────

def calc_i_shape(d: float, bf: float, tf: float, tw: float,
                 k: float = 0.0) -> SectionProps:
    """
    Perfil I/H de alas paralelas simétricas.
    k = altura del filete soldado/laminado (0 si desconocido).
    """
    d  = _pos(d)
    bf = _pos(bf)
    tf = _pos(tf)
    tw = _pos(tw)

    # Altura libre del alma
    h = max(d - 2 * (tf + k), d - 2 * tf, 0.0)

    A = 2.0 * bf * tf + h * tw

    # Eje X–X (fuerte)
    off = d / 2.0 - tf / 2.0
    Ix  = 2.0 * (bf * tf**3 / 12.0 + bf * tf * off**2) + tw * h**3 / 12.0
    Sx  = Ix / (d / 2.0)
    Zx  = 2.0 * (bf * tf * (d / 2.0 - tf / 2.0) + tw * h / 2.0 * h / 4.0)
    rx  = math.sqrt(Ix / _pos(A))

    # Eje Y–Y (débil)
    Iy  = 2.0 * (tf * bf**3 / 12.0) + h * tw**3 / 12.0
    Sy  = Iy / (bf / 2.0)
    Zy  = 2.0 * (tf * (bf / 2.0)**2 / 2.0 + (h / 2.0) * (tw / 2.0)**2 / 2.0)
    ry  = math.sqrt(Iy / _pos(A))

    # Torsión
    J   = (2.0 * bf * tf**3 + h * tw**3) / 3.0
    Cw  = Iy * (d - tf)**2 / 4.0

    # rts (AISC F2-7)
    rts = 0.0
    if Iy > 0 and Cw > 0 and Sx > 0:
        rts = (Iy * Cw)**0.25 / Sx**0.5

    ro = math.sqrt((Ix + Iy) / _pos(A))

    return SectionProps(
        area_mm2=A, Ix_mm4=Ix, Sx_mm3=Sx, Zx_mm3=Zx, rx_mm=rx,
        Iy_mm4=Iy, Sy_mm3=Sy, Zy_mm3=Zy, ry_mm=ry,
        J_mm4=J, Cw_mm6=Cw, rts_mm=rts, ro_mm=ro,
        h_tw=h / _pos(tw),
        bf_2tf=bf / (2.0 * _pos(tf)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Canal U (C, UPN, UPE, MC, IC …)
# ─────────────────────────────────────────────────────────────────────────────

def calc_channel(d: float, bf: float, tf: float, tw: float) -> SectionProps:
    """Canal de ala paralela (no cónica)."""
    d  = _pos(d)
    bf = _pos(bf)
    tf = _pos(tf)
    tw = _pos(tw)

    h = max(d - 2.0 * tf, 0.0)

    A_f = bf * tf
    A_w = h * tw
    A   = 2.0 * A_f + A_w

    # Centroide en x (desde el centro del alma hacia el exterior del ala)
    x_f  = bf / 2.0
    x_w  = tw / 2.0
    x_bar = (2.0 * A_f * x_f + A_w * x_w) / _pos(A)

    # Eje X–X
    off = d / 2.0 - tf / 2.0
    Ix  = 2.0 * (bf * tf**3 / 12.0 + A_f * off**2) + tw * h**3 / 12.0
    Sx  = Ix / (d / 2.0)
    Zx  = 2.0 * (A_f * (d / 2.0 - tf / 2.0) + A_w / 2.0 * (h / 4.0))
    rx  = math.sqrt(Ix / _pos(A))

    # Eje Y–Y (respecto al centroide)
    Iy  = (2.0 * (tf * bf**3 / 12.0 + A_f * (x_f - x_bar)**2)
           + h * tw**3 / 12.0 + A_w * (x_w - x_bar)**2)
    # Módulo elástico: el extremo más alejado del centroide
    y_max = max(x_bar, bf - x_bar)
    Sy   = Iy / _pos(y_max)
    # Módulo plástico (aprox. conservadora)
    Zy   = Iy / _pos(min(x_bar, bf - x_bar))
    ry   = math.sqrt(Iy / _pos(A))

    # Torsión
    J = (2.0 * bf * tf**3 + h * tw**3) / 3.0

    # Centro de corte (AISC Commentary §C-F1 — canal): xo desde centroide
    # xo ≈ bf²·tf·d² / (4·Ix)  (positivo en dirección del ala)
    xo = bf**2 * tf * d**2 / (4.0 * _pos(Ix))

    # Constante de alabeo (Cw)
    Cw = Iy * xo**2 + tf * bf**3 * d**2 / 12.0

    ro = math.sqrt((Ix + Iy + A * xo**2) / _pos(A))

    return SectionProps(
        area_mm2=A, Ix_mm4=Ix, Sx_mm3=Sx, Zx_mm3=Zx, rx_mm=rx,
        Iy_mm4=Iy, Sy_mm3=Sy, Zy_mm3=Zy, ry_mm=ry,
        J_mm4=J, Cw_mm6=Cw, ro_mm=ro, xo_mm=xo,
        h_tw=h / _pos(tw),
        bf_2tf=bf / (2.0 * _pos(tf)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tee (WT, MT, ST)
# ─────────────────────────────────────────────────────────────────────────────

def calc_tee(d: float, bf: float, tf: float, tw: float) -> SectionProps:
    """Perfil T: ala arriba (raíz), alma hacia abajo."""
    d  = _pos(d)
    bf = _pos(bf)
    tf = _pos(tf)
    tw = _pos(tw)

    d_stem = max(d - tf, 0.0)

    A_f = bf * tf
    A_s = d_stem * tw
    A   = A_f + A_s

    # Centroide (y desde la punta del alma, positivo hacia el ala)
    y_f  = d - tf / 2.0
    y_s  = d_stem / 2.0
    y_bar = (A_f * y_f + A_s * y_s) / _pos(A)

    # Eje X–X
    Ix  = (bf * tf**3 / 12.0 + A_f * (y_f - y_bar)**2
           + tw * d_stem**3 / 12.0 + A_s * (y_s - y_bar)**2)
    # Dos módulos elásticos; reportar el menor
    Sx_top = Ix / _pos(d - y_bar)
    Sx_bot = Ix / _pos(y_bar)
    Sx = min(Sx_top, Sx_bot)
    # Módulo plástico (suma de momentos de áreas parciales respecto a E.N.P.)
    Zx = A_f * abs(y_f - y_bar) + A_s * abs(y_s - y_bar)
    rx = math.sqrt(Ix / _pos(A))

    # Eje Y–Y (simétrico)
    Iy  = tf * bf**3 / 12.0 + d_stem * tw**3 / 12.0
    Sy  = Iy / (bf / 2.0)
    Zy  = tf * bf**2 / 4.0 + d_stem * tw**2 / 4.0
    ry  = math.sqrt(Iy / _pos(A))

    J  = (bf * tf**3 + d_stem * tw**3) / 3.0
    Cw = 0.0   # despreciable para T (AISC Commentary)

    ro = math.sqrt((Ix + Iy) / _pos(A))

    return SectionProps(
        area_mm2=A, Ix_mm4=Ix, Sx_mm3=Sx, Zx_mm3=Zx, rx_mm=rx,
        Iy_mm4=Iy, Sy_mm3=Sy, Zy_mm3=Zy, ry_mm=ry,
        J_mm4=J, Cw_mm6=Cw, ro_mm=ro,
        h_tw=d_stem / _pos(tw),
        bf_2tf=bf / (2.0 * _pos(tf)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ángulo L
# ─────────────────────────────────────────────────────────────────────────────

def calc_angle(d: float, b: float, t: float) -> SectionProps:
    """
    Ángulo de piernas d (vertical) y b (horizontal), espesor t.
    Se recomienda d ≥ b (pierna larga vertical).
    """
    d = _pos(d)
    b = _pos(b)
    t = _pos(t)

    # Tres rectángulos (superposición): pierna vertical, pierna horizontal, esquina restada
    A1, A2, A3 = d * t, b * t, t * t
    A = A1 + A2 - A3

    x1, y1 = t / 2.0,  d / 2.0
    x2, y2 = b / 2.0,  t / 2.0
    x3, y3 = t / 2.0,  t / 2.0

    x_bar = (A1 * x1 + A2 * x2 - A3 * x3) / _pos(A)
    y_bar = (A1 * y1 + A2 * y2 - A3 * y3) / _pos(A)

    Ix = (t * d**3 / 12.0 + A1 * (y1 - y_bar)**2
          + b * t**3 / 12.0 + A2 * (y2 - y_bar)**2
          - t**4 / 12.0 - A3 * (y3 - y_bar)**2)

    Iy = (d * t**3 / 12.0 + A1 * (x1 - x_bar)**2
          + t * b**3 / 12.0 + A2 * (x2 - x_bar)**2
          - t**4 / 12.0 - A3 * (x3 - x_bar)**2)

    rx = math.sqrt(Ix / _pos(A))
    ry = math.sqrt(Iy / _pos(A))

    # Módulos elásticos (extremo más alejado del centroide)
    Sx = Ix / _pos(max(d - y_bar, y_bar))
    Sy = Iy / _pos(max(b - x_bar, x_bar))
    Zx = Sx   # conservador: Zx ≈ Sx para ángulo (centroide neutro ≈ elástico)
    Zy = Sy

    J = (d * t**3 + b * t**3 - t**4) / 3.0

    return SectionProps(
        area_mm2=A, Ix_mm4=Ix, Sx_mm3=Sx, Zx_mm3=Zx, rx_mm=rx,
        Iy_mm4=Iy, Sy_mm3=Sy, Zy_mm3=Zy, ry_mm=ry,
        J_mm4=J, Cw_mm6=0.0,
        bf_2tf=b / _pos(t),
    )


# ─────────────────────────────────────────────────────────────────────────────
# HSS rectangular / cajón (CJ, CJE, HSS_R, OC, OCA)
# ─────────────────────────────────────────────────────────────────────────────

def calc_hss_rect(d: float, bf: float, t: float) -> SectionProps:
    """
    HSS rectangular: d=altura exterior, bf=ancho exterior, t=espesor de pared.
    """
    d  = _pos(d)
    bf = _pos(bf)
    t  = _pos(t, 0.1)
    t  = min(t, d / 2.0 - 0.01, bf / 2.0 - 0.01)

    di = max(d  - 2.0 * t, 0.0)
    bi = max(bf - 2.0 * t, 0.0)

    A  = d * bf - di * bi

    # Eje X–X
    Ix = (bf * d**3 - bi * di**3) / 12.0
    Sx = Ix / (d / 2.0)
    Zx = bf * d**2 / 4.0 - bi * di**2 / 4.0
    rx = math.sqrt(Ix / _pos(A))

    # Eje Y–Y
    Iy = (d * bf**3 - di * bi**3) / 12.0
    Sy = Iy / (bf / 2.0)
    Zy = d * bf**2 / 4.0 - di * bi**2 / 4.0
    ry = math.sqrt(Iy / _pos(A))

    # Torsión (Bredt para sección cajón cerrada rectangular)
    # J = 4·Am²·t / Σ(b/t) con Am = área encerrada por línea media
    Am    = (d - t) * (bf - t)
    perim = 2.0 * ((d - t) + (bf - t))
    J     = 4.0 * Am**2 * t / _pos(perim)

    Cw = 0.0   # Cw de cajón cerrado ≈ 0

    ro = math.sqrt((Ix + Iy) / _pos(A))

    return SectionProps(
        area_mm2=A, Ix_mm4=Ix, Sx_mm3=Sx, Zx_mm3=Zx, rx_mm=rx,
        Iy_mm4=Iy, Sy_mm3=Sy, Zy_mm3=Zy, ry_mm=ry,
        J_mm4=J, Cw_mm6=Cw, ro_mm=ro,
        h_tw=(d - 2.0 * t) / _pos(t),
        bf_2tf=(bf - 2.0 * t) / _pos(t),
        D_t=max(d, bf) / _pos(t),
    )


# ─────────────────────────────────────────────────────────────────────────────
# HSS circular (HSS_C, O)
# ─────────────────────────────────────────────────────────────────────────────

def calc_hss_circ(D: float, t: float) -> SectionProps:
    """
    Tubo circular: D=diámetro exterior, t=espesor de pared.
    """
    D = _pos(D)
    t = _pos(t, 0.01)
    t = min(t, D / 2.0 - 0.01)
    Di = D - 2.0 * t

    A  = math.pi / 4.0 * (D**2 - Di**2)
    Ix = math.pi / 64.0 * (D**4 - Di**4)
    Sx = Ix / (D / 2.0)
    Zx = (D**3 - Di**3) / 6.0
    rx = math.sqrt(Ix / _pos(A))

    # Por simetría circular: Y = X
    Iy, Sy, Zy, ry = Ix, Sx, Zx, rx

    J  = 2.0 * Ix   # tubo circular: J = Ip = 2·Ix
    Cw = 0.0

    ro = math.sqrt(2.0 * Ix / _pos(A))

    return SectionProps(
        area_mm2=A, Ix_mm4=Ix, Sx_mm3=Sx, Zx_mm3=Zx, rx_mm=rx,
        Iy_mm4=Iy, Sy_mm3=Sy, Zy_mm3=Zy, ry_mm=ry,
        J_mm4=J, Cw_mm6=Cw, ro_mm=ro,
        D_t=D / _pos(t),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Despacho por family_code
# ─────────────────────────────────────────────────────────────────────────────

def compute_props(section) -> SectionProps:
    """
    Calcula propiedades seccionales desde la geometría del catálogo.
    `section` es una instancia de Section (steeldesigner.catalog.models).
    Devuelve SectionProps sin modificar el objeto de entrada.
    """
    from steeldesigner.core.section_adapter import (
        I_SHAPE_FAMILIES, CHANNEL_FAMILIES, TEE_FAMILIES,
        ANGLE_FAMILIES, HSS_RECT_FAMILIES, HSS_CIRC_FAMILIES,
    )

    def _r(v) -> float:
        return float(v) if v is not None else 0.0

    fc = (section.family.family_code if section.family else "") or ""
    d, bf, tf, tw = _r(section.d), _r(section.bf), _r(section.tf), _r(section.tw)

    if fc in I_SHAPE_FAMILIES:
        if not (d > 0 and bf > 0 and tf > 0 and tw > 0):
            return SectionProps()
        k = _r(getattr(section, "k", None))
        return calc_i_shape(d, bf, tf, tw, k)

    if fc in CHANNEL_FAMILIES:
        if not (d > 0 and bf > 0 and tf > 0 and tw > 0):
            return SectionProps()
        return calc_channel(d, bf, tf, tw)

    if fc in TEE_FAMILIES:
        if not (d > 0 and bf > 0 and tf > 0 and tw > 0):
            return SectionProps()
        return calc_tee(d, bf, tf, tw)

    if fc in ANGLE_FAMILIES:
        b = bf or d
        t = tf or tw
        if not (d > 0 and t > 0):
            return SectionProps()
        return calc_angle(d, b, t)

    if fc in HSS_RECT_FAMILIES:
        t = (_r(getattr(section, "t_des", None))
             or _r(getattr(section, "t_nom", None))
             or tf or tw)
        if not (d > 0 and bf > 0 and t > 0):
            return SectionProps()
        return calc_hss_rect(d, bf, t)

    if fc in HSS_CIRC_FAMILIES:
        t = (_r(getattr(section, "t_des", None))
             or _r(getattr(section, "t_nom", None))
             or tf or tw)
        if not (d > 0 and t > 0):
            return SectionProps()
        return calc_hss_circ(d, t)

    return SectionProps()


def apply_to_section(section) -> None:
    """
    Calcula propiedades desde geometría y las escribe en los campos del
    objeto Section en memoria. No modifica la BD.

    Llamar una vez al inicio de EngineFacade.run() garantiza que todos
    los verificadores usen valores internamente calculados.
    """
    p = compute_props(section)
    if p.area_mm2 <= 0:
        return

    section.area_mm2 = p.area_mm2
    section.Ix_mm4   = p.Ix_mm4
    section.Sx_mm3   = p.Sx_mm3
    section.Zx_mm3   = p.Zx_mm3
    section.rx_mm    = p.rx_mm
    section.Iy_mm4   = p.Iy_mm4
    section.Sy_mm3   = p.Sy_mm3
    section.Zy_mm3   = p.Zy_mm3
    section.ry_mm    = p.ry_mm
    section.J_mm4    = p.J_mm4
    section.Cw_mm6   = p.Cw_mm6
    section.ro_mm    = p.ro_mm
    if p.h_tw    > 0: section.h_tw    = p.h_tw
    if p.bf_2tf  > 0: section.bf_2tf  = p.bf_2tf
    if p.D_t     > 0: section.D_t     = p.D_t
    if p.rts_mm  > 0:
        try:
            section.rts_mm = p.rts_mm
        except AttributeError:
            pass   # campo no existe en versiones antiguas del modelo
    if p.xo_mm != 0:
        try:
            section.xo_mm = p.xo_mm
        except AttributeError:
            pass
