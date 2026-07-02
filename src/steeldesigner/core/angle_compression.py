"""Módulo principal de cálculo para perfiles angulares (L) según AISC 360.

Implementa:
- Propiedades geométricas de secciones L
- Clasificación de sección §B4.1a
- Esbeltez efectiva §E5 (ángulos comprimidos por misma pierna)
- Resistencia nominal a compresión §E3 / §E7
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────
#  DATA CLASSES
# ──────────────────────────────────────────────


@dataclass(slots=True)
class SectionProps:
    """Propiedades geométricas de un perfil angular.

    Nota sobre qs1/qs2/Q: son un indicador preliminar de esbeltez de cada
    pierna (clasificación b/t vs. λr), útil para mostrarlo en pantalla, pero
    NO se usan para reducir Fcr — AISC 360-16/22 §E7 (a diferencia de la
    edición 2010) ya no aplica un factor Q sobre Fy; en su lugar reduce el
    ÁREA efectiva de las piernas esbeltas (ver `compute_capacity`/§E7.1).
    """
    b1: float       # pierna larga (mm)
    b2: float       # pierna corta (mm)
    t: float        # espesor (mm)
    A: float        # área (mm²)
    xc: float       # centroide x desde talón (mm)
    yc: float       # centroide y desde talón (mm)
    Ix: float       # inercia eje x (mm⁴)
    Iy: float       # inercia eje y (mm⁴)
    Ixy: float      # producto de inercia (mm⁴)
    I1: float       # inercia principal máxima (mm⁴)
    I2: float       # inercia principal mínima (mm⁴)
    rx: float       # radio de giro x (mm)
    ry: float       # radio de giro y (mm)
    r1: float       # radio de giro principal máx (mm)
    r2: float       # radio de giro principal mín — rz (mm)
    theta: float    # ángulo x-x → I₁ (rad)
    bt1: float      # b₁/t
    bt2: float      # b₂/t
    lambda_r: float # λr = 0.45√(E/Fy) — límite de esbeltez de pierna, Tabla B4.1a
    qs1: float      # clasificación preliminar pierna larga (informativo, no reduce Fcr)
    qs2: float      # clasificación preliminar pierna corta (informativo, no reduce Fcr)
    Q: float        # min(qs1, qs2) — informativo únicamente, ver nota de clase


@dataclass(slots=True)
class SlendernessResult:
    """Clasificación de esbeltez de una pierna."""
    q: float     # factor de reducción Qs
    cls: str     # "No esbelto", "Esbelto (trans.)", "Esbelto (crít.)"
    ok: bool     # True si b/t ≤ λr


@dataclass(slots=True)
class CapacityResult:
    """Resultado de capacidad a compresión."""
    r_c: float          # radio de giro de la pierna conectada (mm)
    L_r: float          # L / r_c
    eq: str             # "L/r ≤ 80" o "L/r > 80"
    penalty: float      # penalización pierna corta
    KLr_eff: float      # (KL/r) efectivo (antes del piso 0.95·L/rz)
    formula: str        # fórmula aplicada (texto)
    KLr_z: float        # L / rz (eje débil)
    KLr_floor_applied: bool  # True si se aplicó el piso 0.95·L/rz (solo §E5(a)(2), pierna corta)
    KLr_design: float   # esbeltez final usada para Fe/Fn
    Fe: float           # tensión crítica de Euler (MPa)
    Fcr: float          # tensión nominal Fn (MPa) — sin factor Q (AISC 360-16/22 §E3)
    mode: str           # "Inelástico" o "Elástico (Euler)"
    Ae: float           # área efectiva tras reducción por piernas esbeltas (mm²), §E7.1
    leg1_slender: bool  # True si la pierna b1 es esbelta (b/t > límite efectivo)
    leg2_slender: bool  # True si la pierna b2 es esbelta
    Pn: float           # resistencia nominal = Fcr · Ae (N)
    phiPn: float        # resistencia de diseño φcPn (LRFD) o Pn/Ω (ASD) (N)
    method: str         # "LRFD" o "ASD"
    limit_KLr: float    # 4.71√(E/Fy) — límite inelástico/elástico (§E3, sin Q)


# ──────────────────────────────────────────────
#  GEOMETRIC PROPERTIES
# ──────────────────────────────────────────────


def section_props(b1: float, b2: float, t: float,
                   Fy: float = 250.0, E: float = 200000.0) -> SectionProps:
    """Calcula todas las propiedades geométricas y factores de esbeltez.

    Args:
        b1: Pierna larga (mm, se reordena internamente si b1 < b2)
        b2: Pierna corta (mm)
        t:  Espesor (mm)
        Fy: Tensión de fluencia (MPa)
        E:  Módulo de elasticidad (MPa)

    Returns:
        SectionProps con todas las propiedades.
    """
    # Asegurar b1 ≥ b2
    if b1 < b2:
        b1, b2 = b2, b1

    # Áreas parciales
    A1 = b1 * t
    A2 = t * (b2 - t)
    A  = A1 + A2

    # Centroide
    xc = (A1 * b1 / 2 + A2 * t / 2) / A
    yc = (A1 * t / 2 + A2 * (t + (b2 - t) / 2)) / A

    # Momentos de inercia respecto a ejes en talón
    yc2 = t + (b2 - t) / 2
    Ix_h  = b1 * t**3 / 3 + t * (b2 - t)**3 / 12 + A2 * yc2**2
    Iy_h  = t * b1**3 / 3 + (b2 - t) * t**3 / 3
    Ixy_h = A1 * (b1 / 2) * (t / 2) + A2 * (t / 2) * yc2

    # Steiner → ejes centroidales
    Ix  = Ix_h  - A * yc * yc
    Iy  = Iy_h  - A * xc * xc
    Ixy = Ixy_h - A * xc * yc

    # Ejes principales
    avg = (Ix + Iy) / 2
    delta = math.sqrt(((Ix - Iy) / 2)**2 + Ixy**2)
    I1 = avg + delta
    I2 = avg - delta

    # Radios de giro
    rx = math.sqrt(Ix / A)
    ry = math.sqrt(Iy / A)
    r1 = math.sqrt(I1 / A)
    r2 = math.sqrt(I2 / A)  # rz = mínimo

    # Ángulo de ejes principales
    theta = 0.5 * math.atan2(-2 * Ixy, Ix - Iy)

    # Relaciones ancho/espesor
    bt1 = b1 / t
    bt2 = b2 / t
    lambda_r = 0.45 * math.sqrt(E / Fy)

    # Factor de esbeltez Qs por pierna
    qs1 = _qs_factor(bt1, Fy, E)
    qs2 = _qs_factor(bt2, Fy, E)
    Q = min(qs1.q, qs2.q)

    return SectionProps(
        b1=b1, b2=b2, t=t, A=A, xc=xc, yc=yc,
        Ix=Ix, Iy=Iy, Ixy=Ixy, I1=I1, I2=I2,
        rx=rx, ry=ry, r1=r1, r2=r2, theta=theta,
        bt1=bt1, bt2=bt2, lambda_r=lambda_r,
        qs1=qs1.q, qs2=qs2.q, Q=Q,
    )


def _qs_factor(bt: float, Fy: float, E: float) -> SlendernessResult:
    """Factor de reducción Qs para piernas sobresalientes (§E7.1a)."""
    l1 = 0.45 * math.sqrt(E / Fy)   # límite no esbelto
    l2 = 0.91 * math.sqrt(E / Fy)   # límite crítico

    if bt <= l1:
        return SlendernessResult(q=1.0, cls="No esbelto", ok=True)
    elif bt <= l2:
        q = 1.34 - 0.76 * bt * math.sqrt(Fy / E)
        return SlendernessResult(q=round(q, 4), cls="Esbelto (trans.)", ok=False)
    else:
        q = 0.53 * E / (Fy * bt**2)
        return SlendernessResult(q=round(q, 4), cls="Esbelto (crít.)", ok=False)


# ──────────────────────────────────────────────
#  CAPACITY (§E5 + §E3/§E7)
# ──────────────────────────────────────────────


def _effective_width(b: float, t: float, Fy: float, E: float, Fn: float) -> tuple[float, bool]:
    """Ancho efectivo de una pierna de ángulo (elemento no rigidizado), §E7.1.

    AISC 360-16/22 reemplazó el método de factor Q (edición 2010) por este
    método de área efectiva: se compara el b/t contra un límite que depende
    de Fn (no solo de Fy), y si es esbelto se reduce el ancho de esa pierna
    en vez de reducir la tensión admisible.

    Coeficientes c1=0.22, c2=1.49 corresponden al caso (c) "todos los demás
    elementos" de la Tabla E7.1 (piernas de ángulos simples). λr=0.45√(E/Fy)
    es el límite de esbeltez de pierna de ángulo de la Tabla B4.1a.

    Returns:
        (b_efectivo, es_esbelta)
    """
    lam = b / t
    lam_r = 0.45 * math.sqrt(E / Fy)
    c1, c2 = 0.22, 1.49

    if lam <= lam_r * math.sqrt(Fy / Fn):
        return b, False

    Fel = (c2 * lam_r / lam) ** 2 * Fy
    be = b * (1 - c1 * math.sqrt(Fel / Fn)) * math.sqrt(Fel / Fn)
    return be, True


def compute_capacity(sec: SectionProps, L: float,
                     conn_leg: str = "long",
                     Fy: float = 250.0, E: float = 200000.0,
                     method: str = "LRFD") -> CapacityResult:
    """Calcula la capacidad a compresión de un ángulo simple.

    Args:
        sec:    Propiedades de la sección (SectionProps).
        L:      Longitud libre (mm).
        conn_leg: Pierna conectada: "long" (b₁) o "short" (b₂).
        Fy:     Tensión de fluencia (MPa).
        E:      Módulo de elasticidad (MPa).

    Returns:
        CapacityResult con todos los valores.
    """
    # Radio de giro de la pierna conectada
    r_c = sec.rx if conn_leg == "long" else sec.ry
    L_r = L / r_c
    eq = "L/r ≤ 80" if L_r <= 80 else "L/r > 80"

    # Penalización por conexión en pierna corta, ángulos de lados desiguales
    # (§E5(a)(2))
    unequal_short = conn_leg == "short" and sec.b1 != sec.b2
    penalty = 0.0
    if unequal_short:
        penalty = 4 * ((sec.b1 / sec.b2)**2 - 1)

    # Esbeltez efectiva (KL/r)eff — §E5(a)
    if L_r <= 80:
        KLr_eff = 72 + 0.75 * L_r + penalty
        formula = f"72 + 0.75·{L_r:.1f}"
    else:
        KLr_eff = 32 + 1.25 * L_r + penalty
        formula = f"32 + 1.25·{L_r:.1f}"

    if penalty > 0:
        formula += f" + {penalty:.2f} (penalización)"

    KLr_eff = min(KLr_eff, 200.0)  # tope máximo, §E5(4)

    # Esbeltez respecto al eje débil z (L/rz)
    KLr_z = L / sec.r2

    # §E5(a)(2): SOLO para ángulos de lados desiguales conectados por la
    # pierna corta, (KL/r) no puede ser menor que 0.95·L/rz. Para pierna
    # larga (o lados iguales) la norma no exige este piso.
    KLr_floor_applied = False
    if unequal_short:
        floor = 0.95 * KLr_z
        if floor > KLr_eff:
            KLr_design = floor
            KLr_floor_applied = True
        else:
            KLr_design = KLr_eff
    else:
        KLr_design = KLr_eff

    # Tensión crítica de Euler
    Fe = math.pi**2 * E / KLr_design**2

    # Límite entre inelástico y elástico, §E3 (sin factor Q — ver nota en
    # SectionProps: la edición 2016/2022 ya no aplica Q a Fy/Fe).
    limit_KLr = 4.71 * math.sqrt(E / Fy)

    # Tensión nominal Fn, §E3 (idéntica para ambos casos, sin Q)
    if KLr_design <= limit_KLr:
        Fcr = (0.658 ** (Fy / Fe)) * Fy
        mode = "Inelástico"
    else:
        Fcr = 0.877 * Fe
        mode = "Elástico (Euler)"

    # Reducción de área por piernas esbeltas, §E7.1 (reemplaza el factor Q)
    be1, leg1_slender = _effective_width(sec.b1, sec.t, Fy, E, Fcr)
    be2, leg2_slender = _effective_width(sec.b2, sec.t, Fy, E, Fcr)
    Ae = sec.A - (sec.b1 - be1) * sec.t - (sec.b2 - be2) * sec.t

    # Resistencia
    Pn = Fcr * Ae       # N
    if method == "LRFD":
        phiPn = 0.90 * Pn       # φc = 0.90
    else:
        phiPn = Pn / 1.67       # Ωc = 1.67

    return CapacityResult(
        r_c=r_c, L_r=L_r, eq=eq, penalty=penalty,
        KLr_eff=KLr_eff, formula=formula, KLr_z=KLr_z,
        KLr_floor_applied=KLr_floor_applied, KLr_design=KLr_design, Fe=Fe,
        Fcr=Fcr, mode=mode, Ae=Ae, leg1_slender=leg1_slender, leg2_slender=leg2_slender,
        Pn=Pn, phiPn=phiPn, method=method, limit_KLr=limit_KLr,
    )


# ──────────────────────────────────────────────
#  CONVENIENCE
# ──────────────────────────────────────────────


def build_calc_steps(sec: SectionProps, cap: CapacityResult,
                      Pu: float, Fy: float, E: float) -> list[dict]:
    """Arma la secuencia de pasos de cálculo (fórmula + sustitución + resultado).

    Reutilizada tanto por la tabla compacta de la GUI como por el informe HTML,
    para no duplicar el texto del informe por fuera del cálculo real.
    """
    phiPn_kN = cap.phiPn / 1000
    ratio = Pu / phiPn_kN if phiPn_kN > 0 else float("inf")
    phi_label = "φcPn" if cap.method == "LRFD" else "Pn/Ωc"
    phi_formula = "φc·Pn (φc=0.90)" if cap.method == "LRFD" else "Pn/Ωc (Ωc=1.67)"
    phi_sub = f"0.90 × {cap.Pn/1000:.2f}" if cap.method == "LRFD" else f"{cap.Pn/1000:.2f} / 1.67"

    return [
        {
            "label": "Esbeltez de la pierna conectada (L/r)",
            "formula": "L / r_c",
            "substitution": f"{cap.eq.split()[0]} → r_c={cap.r_c:.2f} mm",
            "result": f"{cap.L_r:.1f}",
            "unit": "",
            "ref": "AISC 360 §E5",
        },
        {
            "label": "Esbeltez efectiva (KL/r)eff",
            "formula": cap.formula.split(" +")[0] if "+" in cap.formula else cap.formula,
            "substitution": cap.formula,
            "result": f"{cap.KLr_eff:.1f}",
            "unit": "",
            "ref": "AISC 360 §E5(a)/(b)",
        },
        {
            "label": "Esbeltez eje débil (L/rz)" + (" — piso 0.95·L/rz aplicado" if cap.KLr_floor_applied else ""),
            "formula": "0.95 · L / r_z" if cap.KLr_floor_applied else "L / r_z (referencial, no gobierna)",
            "substitution": (
                f"0.95 × {cap.KLr_z * sec.r2:.0f} / {sec.r2:.2f}" if cap.KLr_floor_applied
                else f"L / {sec.r2:.2f}"
            ),
            "result": f"{cap.KLr_z:.1f}",
            "unit": "",
            "ref": "AISC 360 §E5(a)(2) — solo aplica a lados desiguales, pierna corta",
        },
        {
            "label": "Esbeltez de diseño (KL/r)design",
            "formula": "max[(KL/r)eff, 0.95·L/rz]" if cap.KLr_floor_applied else "(KL/r)eff",
            "substitution": (
                f"max[{cap.KLr_eff:.1f}, 0.95×{cap.KLr_z:.1f}]" if cap.KLr_floor_applied
                else f"{cap.KLr_eff:.1f}"
            ),
            "result": f"{cap.KLr_design:.1f}",
            "unit": "",
            "ref": "AISC 360 §E5(a)(2)",
        },
        {
            "label": "Tensión crítica de Euler Fe",
            "formula": "π²E / (KL/r)design²",
            "substitution": f"π²×{E:.0f} / {cap.KLr_design:.1f}²",
            "result": f"{cap.Fe:.1f}",
            "unit": "MPa",
            "ref": "AISC 360 §E3",
        },
        {
            "label": f"Tensión nominal Fn ({cap.mode})",
            "formula": "0.658^(Fy/Fe)·Fy" if cap.mode == "Inelástico" else "0.877·Fe",
            "substitution": (
                f"0.658^({Fy:.0f}/{cap.Fe:.1f})×{Fy:.0f}"
                if cap.mode == "Inelástico" else f"0.877×{cap.Fe:.1f}"
            ),
            "result": f"{cap.Fcr:.1f}",
            "unit": "MPa",
            "ref": "AISC 360 §E3",
        },
        {
            "label": "Área efectiva Ae" + (
                " (piernas no esbeltas, Ae=Ag)" if not (cap.leg1_slender or cap.leg2_slender) else ""
            ),
            "formula": "Ag − Σ(b − be)·t",
            "substitution": f"{sec.A:.1f} − reducción" if (cap.leg1_slender or cap.leg2_slender) else f"{sec.A:.1f}",
            "result": f"{cap.Ae:.1f}",
            "unit": "mm²",
            "ref": "AISC 360 §E7.1",
        },
        {
            "label": "Resistencia nominal Pn",
            "formula": "Fn · Ae",
            "substitution": f"{cap.Fcr:.1f} × {cap.Ae:.1f}",
            "result": f"{cap.Pn/1000:.2f}",
            "unit": "kN",
            "ref": "AISC 360 §E7-1" if (cap.leg1_slender or cap.leg2_slender) else "AISC 360 §E3-1",
        },
        {
            "label": f"Resistencia de diseño {phi_label}",
            "formula": phi_formula,
            "substitution": phi_sub,
            "result": f"{phiPn_kN:.2f}",
            "unit": "kN",
            "ref": "AISC 360 §E1",
        },
        {
            "label": "Relación demanda/capacidad D/C",
            "formula": f"Pu / {phi_label}",
            "substitution": f"{Pu:.2f} / {phiPn_kN:.2f}",
            "result": f"{ratio:.3f}",
            "unit": "",
            "ref": "",
        },
    ]


def check_angle(b1: float, b2: float, t: float,
                L: float, Pu: float,
                Fy: float = 250.0, E: float = 200000.0,
                conn_leg: str = "long",
                method: str = "LRFD") -> dict:
    """Verifica un ángulo a compresión. Retorna dict con todo.

    Args:
        b1: Pierna larga (mm)
        b2: Pierna corta (mm)
        t:  Espesor (mm)
        L:  Longitud libre (mm)
        Pu: Carga axial mayorada (kN)
        Fy: Fluencia (MPa)
        E:  Módulo elástico (MPa)
        conn_leg: "long" o "short"

    Returns:
        Dict con resultados completos para reporte/UI.
    """
    sec = section_props(b1, b2, t, Fy, E)
    cap = compute_capacity(sec, L, conn_leg, Fy, E, method)

    phiPn_kN = cap.phiPn / 1000
    ratio = Pu / phiPn_kN if phiPn_kN > 0 else float('inf')
    ok = ratio <= 1.0
    calc_steps = build_calc_steps(sec, cap, Pu, Fy, E)

    return {
        "ok": ok,
        "ratio": ratio,
        "reserve_pct": (1 - ratio) * 100 if ok else 0,
        "deficit_pct": (ratio - 1) * 100 if not ok else 0,
        "Pu_kN": Pu,
        "phiPn_kN": round(phiPn_kN, 2),
        "Pn_kN": round(cap.Pn / 1000, 2),
        "Fcr_MPa": round(cap.Fcr, 2),
        "mode": cap.mode,
        "(KL/r)eff": round(cap.KLr_eff, 1),
        "L/r": round(cap.L_r, 1),
        "L/rz": round(cap.KLr_z, 1),
        "(KL/r)design": round(cap.KLr_design, 1),
        "KLr_floor_applied": cap.KLr_floor_applied,
        "Fe_MPa": round(cap.Fe, 1),
        "Q": sec.Q,
        "b1/t": round(sec.bt1, 2),
        "b2/t": round(sec.bt2, 2),
        "lambda_r": round(sec.lambda_r, 2),
        "A_mm2": round(sec.A, 1),
        "Ae_mm2": round(cap.Ae, 1),
        "leg1_slender": cap.leg1_slender,
        "leg2_slender": cap.leg2_slender,
        "rx_mm": round(sec.rx, 2),
        "ry_mm": round(sec.ry, 2),
        "rz_mm": round(sec.r2, 2),
        "Ix_cm4": round(sec.Ix / 1e4, 2),
        "Iy_cm4": round(sec.Iy / 1e4, 2),
        "section": f"L{sec.b1:.0f}x{sec.b2:.0f}x{sec.t:.0f}",
        "conn_leg": conn_leg,
        "method": method,
        "Fy_MPa": Fy,
        "E_GPa": E / 1000,
        "formula": cap.formula,
        "calc_steps": calc_steps,
    }
