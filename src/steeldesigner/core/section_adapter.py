"""
Adaptador entre Section del catálogo y los dataclasses del motor AISC 360.

Convierte Section (todo en mm/mm⁴/MPa) a ISection, ChannelSection,
TeeSection, o argumentos de angle_compression.check_angle().
"""
from __future__ import annotations

from steeldesigner.catalog.models import Section
from steeldesigner.core.aisc360_engine import (
    ISection, ChannelSection, TeeSection, AngleSection, Material as EngineMaterial
)


# Familias por tipo de motor
I_SHAPE_FAMILIES = {
    "W", "HP", "H", "IN", "HN", "IP", "IE", "PH", "HR",
    "IPE", "IPN", "HEA", "HEB", "HEM", "HL", "HD",
}
CHANNEL_FAMILIES = {"C", "CA", "MC", "UPN", "UPE", "IC", "ICA"}
TEE_FAMILIES = {"WT", "MT", "ST", "T"}
ANGLE_FAMILIES = {"L_ICHA_PLEG", "L_ICHA_LAM", "L_AISC", "L"}
HSS_RECT_FAMILIES = {"CJ", "CJE", "HSS_R", "OC", "OCA"}
HSS_CIRC_FAMILIES = {"HSS_C", "O"}


def family_type(sec: Section) -> str:
    code = (sec.family.family_code if sec.family else "") or ""
    if code in I_SHAPE_FAMILIES:
        return "i_shape"
    if code in CHANNEL_FAMILIES:
        return "channel"
    if code in TEE_FAMILIES:
        return "tee"
    if code in ANGLE_FAMILIES:
        return "angle"
    if code in HSS_RECT_FAMILIES:
        return "hss_rect"
    if code in HSS_CIRC_FAMILIES:
        return "hss_circ"
    return "unknown"


def _safe(val, default: float = 0.0) -> float:
    return float(val) if val is not None else default


def to_engine_material(Fy_MPa: float, Fu_MPa: float,
                        E_MPa: float = 200_000.0, G_MPa: float = 77_200.0) -> EngineMaterial:
    return EngineMaterial(Fy=Fy_MPa, Fu=Fu_MPa, E=E_MPa, G=G_MPa)


def to_isection(sec: Section) -> ISection:
    d = _safe(sec.d)
    bf = _safe(sec.bf)
    tf = _safe(sec.tf)
    tw = _safe(sec.tw)
    A = _safe(sec.area_mm2)
    Ix = _safe(sec.Ix_mm4)
    Iy = _safe(sec.Iy_mm4)
    Sx = _safe(sec.Sx_mm3)
    Sy = _safe(sec.Sy_mm3)
    Zx = _safe(sec.Zx_mm3)
    Zy = _safe(sec.Zy_mm3)
    rx = _safe(sec.rx_mm)
    ry = _safe(sec.ry_mm)
    J = _safe(sec.J_mm4)
    Cw = _safe(sec.Cw_mm6)
    ro = _safe(sec.ro_mm)

    # rts: radio de giro del ala comprimida + 1/3 del alma (AISC F2)
    rts = _safe(getattr(sec, "rts_mm", None))
    if rts == 0.0 and Iy > 0 and Cw > 0 and Sx > 0:
        import math
        rts = (Iy * Cw) ** 0.25 / Sx ** 0.5

    return ISection(
        name=sec.designation_modern or sec.designation_legacy or "Unknown",
        area=A, d=d, bf=bf, tf=tf, tw=tw,
        Ix=Ix, Iy=Iy, Zx=Zx, Zy=Zy, Sx=Sx, Sy=Sy,
        rx=rx, ry=ry, J=J, Cw=Cw, ro=ro, rts=rts,
        x_sc=_safe(sec.xo_mm), y_sc=_safe(sec.io_mm),
    )


def to_channel_section(sec: Section) -> ChannelSection:
    return ChannelSection(
        name=sec.designation_modern or sec.designation_legacy or "Unknown",
        area=_safe(sec.area_mm2),
        d=_safe(sec.d), bf=_safe(sec.bf), tf=_safe(sec.tf), tw=_safe(sec.tw),
        Ix=_safe(sec.Ix_mm4), Iy=_safe(sec.Iy_mm4),
        Zx=_safe(sec.Zx_mm3), Zy=_safe(sec.Zy_mm3),
        Sx=_safe(sec.Sx_mm3), Sy=_safe(sec.Sy_mm3),
        rx=_safe(sec.rx_mm), ry=_safe(sec.ry_mm),
        J=_safe(sec.J_mm4), Cw=_safe(sec.Cw_mm6),
        ro=_safe(sec.ro_mm),
        x_sc=_safe(sec.xo_mm),
    )


def to_tee_section(sec: Section) -> TeeSection:
    return TeeSection(
        name=sec.designation_modern or sec.designation_legacy or "Unknown",
        area=_safe(sec.area_mm2),
        d=_safe(sec.d), bf=_safe(sec.bf), tf=_safe(sec.tf), tw=_safe(sec.tw),
        Ix=_safe(sec.Ix_mm4), Iy=_safe(sec.Iy_mm4),
        Sx=_safe(sec.Sx_mm3), Sy=_safe(sec.Sy_mm3),
        Zx=_safe(sec.Zx_mm3), Zy=_safe(sec.Zy_mm3),
        rx=_safe(sec.rx_mm), ry=_safe(sec.ry_mm),
        J=_safe(sec.J_mm4), Cw=_safe(sec.Cw_mm6),
        ro=_safe(sec.ro_mm),
    )


def to_angle_section(sec: Section) -> AngleSection:
    """Para el motor genérico (AngleMember). Para §E5 usa to_angle_args()."""
    d = _safe(sec.d)
    b = _safe(sec.bf) or d  # igual si no tiene bf
    t = _safe(sec.tf) or _safe(sec.tw)
    return AngleSection(
        name=sec.designation_modern or sec.designation_legacy or "Unknown",
        area=_safe(sec.area_mm2),
        d=d, b=b, t=t,
        Ix=_safe(sec.Ix_mm4), Iy=_safe(sec.Iy_mm4),
        rx=_safe(sec.rx_mm), ry=_safe(sec.ry_mm),
        Sx=_safe(sec.Sx_mm3), Sy=_safe(sec.Sy_mm3),
        Zx=_safe(sec.Zx_mm3), Zy=_safe(sec.Zy_mm3),
        J=_safe(sec.J_mm4),
    )


def to_angle_args(sec: Section) -> dict:
    """Argumentos para angle_compression.check_angle() — §E5 correcto."""
    d = _safe(sec.d)
    b = _safe(sec.bf)
    if b == 0:
        b = d  # ángulo igual
    t = _safe(sec.tf) or _safe(sec.tw)
    # b1 = pierna larga, b2 = pierna corta
    b1, b2 = (d, b) if d >= b else (b, d)
    return {"b1": b1, "b2": b2, "t": t}


def section_props_dict(sec: Section) -> dict:
    """Propiedades resumidas para mostrar en UI."""
    return {
        "Designation": sec.designation_modern or sec.designation_legacy,
        "Family": sec.family.family_code if sec.family else "",
        "d (mm)": _safe(sec.d),
        "bf (mm)": _safe(sec.bf),
        "tf (mm)": _safe(sec.tf),
        "tw (mm)": _safe(sec.tw),
        "A (mm²)": _safe(sec.area_mm2),
        "Weight (kg/m)": _safe(sec.weight_kg_m),
        "Ix (mm⁴)": _safe(sec.Ix_mm4),
        "Sx (mm³)": _safe(sec.Sx_mm3),
        "Zx (mm³)": _safe(sec.Zx_mm3),
        "rx (mm)": _safe(sec.rx_mm),
        "Iy (mm⁴)": _safe(sec.Iy_mm4),
        "Sy (mm³)": _safe(sec.Sy_mm3),
        "Zy (mm³)": _safe(sec.Zy_mm3),
        "ry (mm)": _safe(sec.ry_mm),
        "J (mm⁴)": _safe(sec.J_mm4),
        "Cw (mm⁶)": _safe(sec.Cw_mm6),
        "ro (mm)": _safe(sec.ro_mm),
    }
