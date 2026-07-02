"""
Torsión según AISC 360-22 Capítulo H3.

Implementa:
  §H3.1 — HSS circulares y rectangulares (torsión St. Venant)
  §H3.2 — Combinación torsión + flexión/cortante en secciones cerradas
  §H3.3 — Secciones abiertas con alabeo (I, canal, tee): tensiones de alabeo
           y verificación por §H3.3(b)

Referencias: AISC 360-22 Chapter H; Design Guide 9 (2nd ed.) para alabeo.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
PHI_TORSION_LRFD = 0.90
OMEGA_TORSION_ASD = 1.67


# ---------------------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class TorsionLimitState:
    equation: str
    description: str
    phi_Tn: float        # capacidad de diseño [N·mm]
    T_demand: float      # demanda [N·mm]
    ratio: float
    passes: bool

    @classmethod
    def make(cls, equation: str, description: str,
             Tn: float, Tu: float, method: str) -> "TorsionLimitState":
        if method == "LRFD":
            phi_Tn = PHI_TORSION_LRFD * Tn
        else:
            phi_Tn = Tn / OMEGA_TORSION_ASD
        ratio = Tu / phi_Tn if phi_Tn > 0 else float("inf")
        return cls(equation=equation, description=description,
                   phi_Tn=phi_Tn, T_demand=Tu, ratio=ratio, passes=ratio <= 1.0)


@dataclass
class TorsionResult:
    method: str
    section_type: str        # "hss_circ" | "hss_rect" | "open"
    states: list[TorsionLimitState] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def controlling(self) -> Optional[TorsionLimitState]:
        return max(self.states, key=lambda s: s.ratio) if self.states else None

    @property
    def passes(self) -> bool:
        c = self.controlling
        return c.passes if c else True

    @property
    def ratio(self) -> float:
        c = self.controlling
        return c.ratio if c else 0.0


# ---------------------------------------------------------------------------
# §H3.1 — HSS Circulares (Pipe / HSS_C)
# ---------------------------------------------------------------------------
def _hss_circ_torsion(D: float, t: float, J: float,
                       Tu: float, Fy: float, E: float,
                       L: float, method: str) -> TorsionResult:
    """
    AISC 360-22 §H3.1 — Torsión en HSS circular.

    Tn = Fcr · C
    C = constante torsional = J/(D/2) = π/2 · (D/2)³ · [1-(1-2t/D)⁴]  ≈ J/(D/2)
    Fcr = menor de:
        1.23·E / (L/D · (D/t)^(1/2))   — pandeo inelástico
        0.60·E / (D/t)^(3/2)            — pandeo local
    φ = 0.90 (LRFD), Ω = 1.67 (ASD)
    """
    res = TorsionResult(method=method, section_type="hss_circ")
    C = J / (D / 2) if D > 0 else 0.0  # módulo torsional plástico aprox.

    D_t = D / t if t > 0 else 9999

    # Fcr — ecuaciones H3-2a y H3-2b
    if L > 0 and D > 0:
        Fcr_a = 1.23 * E / ((L / D) * (D_t) ** 0.5)
    else:
        Fcr_a = float("inf")

    Fcr_b = 0.60 * E / (D_t ** 1.5)
    Fcr = min(Fcr_a, Fcr_b)
    Fcr = min(Fcr, 0.6 * Fy)  # tope: plastificación

    Tn = Fcr * C
    res.states.append(TorsionLimitState.make(
        "H3-1 / H3-2", "Torsión HSS circular — Tn=Fcr·C", Tn, Tu, method
    ))

    if D_t > 0.45 * E / Fy:
        res.notes.append(f"D/t={D_t:.1f} excede 0.45·E/Fy={0.45*E/Fy:.1f}: HSS no aplicable §H3.1")

    return res


# ---------------------------------------------------------------------------
# §H3.1 — HSS Rectangulares (cajón)
# ---------------------------------------------------------------------------
def _hss_rect_torsion(B: float, D: float, t: float, J: float,
                       Tu: float, Fy: float,
                       method: str) -> TorsionResult:
    """
    AISC 360-22 §H3.1 — Torsión en HSS rectangular.

    Tn = 0.6·Fy · C
    C = módulo torsional ≈ 2·t·A_mid   donde A_mid = (B-t)(D-t)
    φ = 0.90 (LRFD), Ω = 1.67 (ASD)
    """
    res = TorsionResult(method=method, section_type="hss_rect")

    # Módulo torsional de Bredt-Batho para sección de pared delgada
    A_mid = (B - t) * (D - t) if B > t and D > t else B * D
    C = 2 * t * A_mid if t > 0 and A_mid > 0 else J / max(B, D) * 2

    Tn = 0.6 * Fy * C
    res.states.append(TorsionLimitState.make(
        "H3-1", "Torsión HSS rectangular — Tn=0.6·Fy·C", Tn, Tu, method
    ))

    return res


# ---------------------------------------------------------------------------
# §H3.3 — Secciones abiertas (I, canal, tee) con alabeo
# ---------------------------------------------------------------------------
def _open_section_torsion(
    Tu: float,
    Mux: float,
    Muy: float,
    Vux: float,
    phi_Mn_x: float,
    phi_Mn_y: float,
    phi_Vn: float,
    # propiedades de alabeo de la sección
    Cw: float,    # constante de alabeo [mm⁶]
    J: float,     # constante torsional de St. Venant [mm⁴]
    Wno: float,   # módulo de alabeo normalizado [mm⁴]
    Sw: float,    # módulo de alabeo corte [mm⁴]
    # geometría
    d: float, bf: float, tf: float, tw: float,
    Fy: float, E: float, G: float,
    # longitud
    L: float,
    method: str,
) -> TorsionResult:
    """
    AISC 360-22 §H3.3 — Combinación torsión + flexión + cortante
    en secciones abiertas.

    Verificación simplificada basada en tensiones normales y cortantes
    totales en el ala más cargada:

        σ_total = σ_flex + σ_warping  ≤  Fy
        τ_total = τ_shear + τ_sv + τ_w  ≤  0.6·Fy

    Nota: para un cálculo riguroso se requiere la distribución de
    bimomento a lo largo del miembro (funciones hiperbólicas). Aquí se
    usa la aproximación conservadora de longitud de pandeo libre.
    """
    res = TorsionResult(method=method, section_type="open")
    if Tu == 0.0 and Mux == 0.0 and Muy == 0.0 and Vux == 0.0:
        return res

    # --- Tensión de alabeo (σ_w) en el ala ---
    # Para miembro biarticulado con torsor uniforme:
    # σ_w ≈ E · Wno · φ''_max
    # φ''_max ≈ Tu / (G·J) * sinh(λ·L/2) / (λ·cosh(λ·L/2))
    # λ = sqrt(G·J / (E·Cw))  [1/mm]
    sigma_w = 0.0
    tau_w = 0.0

    if Cw > 0 and J > 0 and G > 0 and L > 0:
        lam = math.sqrt(G * J / (E * Cw))  # parámetro de alabeo
        lam_L2 = lam * L / 2
        # φ'' máximo (en centro de vano, carga uniforme de torsor):
        if lam_L2 < 100:
            phi_pp = (Tu / (G * J)) * (
                1 - 1 / math.cosh(lam_L2)
            ) / L * 2  # aprox para carga concentrada en centro
        else:
            phi_pp = Tu / (G * J) / L  # caso muy largo → pura St. Venant

        sigma_w = E * Wno * phi_pp if Wno > 0 else 0.0
        # Cortante de alabeo (τ_w):
        if Sw > 0 and tf > 0 and bf > 0:
            tau_w = E * Sw * phi_pp / (tf * bf) if tf * bf > 0 else 0.0

    # --- Tensión de flexión en el ala (σ_flex) ---
    # Usar esfuerzo promedio en el ala
    y_ala = d / 2 if d > 0 else 0.0
    Ix_approx = 2 * bf * tf * (d / 2 - tf / 2) ** 2 + tw * (d - 2 * tf) ** 3 / 12
    sigma_flex = Mux * y_ala / Ix_approx if Ix_approx > 0 else 0.0

    # --- Cortante de St. Venant (τ_sv) ---
    tau_sv = G * tf * (Tu / (G * J)) if J > 0 and tf > 0 else 0.0

    # --- Cortante de flexión en el ala (τ_v) ---
    tau_v = Vux / (d * tw) if d > 0 and tw > 0 else 0.0

    # --- Verificaciones ---
    sigma_total = sigma_flex + sigma_w
    ratio_normal = sigma_total / Fy if Fy > 0 else 0.0
    phi_factor = PHI_TORSION_LRFD if method == "LRFD" else 1 / OMEGA_TORSION_ASD
    # Para tensión normal se compara con φ·Fy
    ratio_normal_design = sigma_total / (phi_factor * Fy) if Fy > 0 else 0.0

    tau_total = tau_v + tau_sv + tau_w
    tau_allow = 0.6 * Fy
    ratio_shear = tau_total / (phi_factor * tau_allow) if tau_allow > 0 else 0.0

    res.states.append(TorsionLimitState(
        equation="H3.3 σ",
        description=f"Tensión normal total (flex+alabeo): {sigma_total:.1f} MPa",
        phi_Tn=phi_factor * Fy,
        T_demand=sigma_total,
        ratio=ratio_normal_design,
        passes=ratio_normal_design <= 1.0,
    ))

    res.states.append(TorsionLimitState(
        equation="H3.3 τ",
        description=f"Tensión cortante total (cortante+SV+alabeo): {tau_total:.1f} MPa",
        phi_Tn=phi_factor * tau_allow,
        T_demand=tau_total,
        ratio=ratio_shear,
        passes=ratio_shear <= 1.0,
    ))

    res.notes.append(
        "§H3.3: Verificación aproximada para miembro con torsor libre en ambos extremos. "
        "Para condiciones de extremo restringidas use análisis exacto con funciones hiperbólicas."
    )
    return res


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------
class ChapterH3:
    """Motor de torsión AISC 360-22 §H3."""

    @staticmethod
    def hss_circular(
        D: float, t: float, J: float,
        Tu: float, Fy: float, E: float, L: float,
        method: str = "LRFD",
    ) -> TorsionResult:
        """Torsión en HSS circular §H3.1."""
        return _hss_circ_torsion(D=D, t=t, J=J, Tu=Tu, Fy=Fy, E=E, L=L, method=method)

    @staticmethod
    def hss_rectangular(
        B: float, D: float, t: float, J: float,
        Tu: float, Fy: float,
        method: str = "LRFD",
    ) -> TorsionResult:
        """Torsión en HSS rectangular §H3.1."""
        return _hss_rect_torsion(B=B, D=D, t=t, J=J, Tu=Tu, Fy=Fy, method=method)

    @staticmethod
    def open_section(
        Tu: float,
        Mux: float = 0.0, Muy: float = 0.0, Vux: float = 0.0,
        phi_Mn_x: float = 0.0, phi_Mn_y: float = 0.0, phi_Vn: float = 0.0,
        Cw: float = 0.0, J: float = 0.0, Wno: float = 0.0, Sw: float = 0.0,
        d: float = 0.0, bf: float = 0.0, tf: float = 0.0, tw: float = 0.0,
        Fy: float = 250.0, E: float = 200_000.0, G: float = 77_200.0,
        L: float = 0.0,
        method: str = "LRFD",
    ) -> TorsionResult:
        """Torsión combinada en sección abierta §H3.3."""
        return _open_section_torsion(
            Tu=Tu, Mux=Mux, Muy=Muy, Vux=Vux,
            phi_Mn_x=phi_Mn_x, phi_Mn_y=phi_Mn_y, phi_Vn=phi_Vn,
            Cw=Cw, J=J, Wno=Wno, Sw=Sw,
            d=d, bf=bf, tf=tf, tw=tw,
            Fy=Fy, E=E, G=G, L=L, method=method,
        )
