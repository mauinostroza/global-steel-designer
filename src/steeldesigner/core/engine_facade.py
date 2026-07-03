"""
Fachada unificada del motor de diseño AISC 360-22.

Enruta cada perfil al motor correcto según su familia:
  - I shapes (W, IN, HN, IPE, HEA, …) → aisc360_engine §D/E/F/G/H
  - Canales (C, MC, UPN, …)            → aisc360_engine §D/E/F/G/H
  - Tees (WT, MT, ST, T)               → aisc360_engine §D/E/F/G/H
  - Ángulos (L_ICHA, L_AISC, …)        → angle_compression §E5 correcto
  - HSS rect. (CJ, HSS_R, …)           → aisc360_engine E7 + torsion_chapter_h3
  - HSS circ. (HSS_C, O)               → aisc360_engine E7 + torsion_chapter_h3
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from steeldesigner.catalog.models import Section
from steeldesigner.core.section_adapter import (
    family_type, _safe,
    to_isection, to_channel_section, to_tee_section,
    to_angle_section, to_angle_args,
)
from steeldesigner.core.aisc360_engine import (
    IShapeMember, ChannelMember, AngleMember, TeeMember,
    Material as EngineMaterial,
    MemberLengths, MemberDemand,
    FlexureInput, ShearInput, EffectiveAreaInput, BlockShearInput,
    DesignMethod, Strictness,
)
from steeldesigner.core.aisc360_master_engine import MasterEngineV2
from steeldesigner.core.angle_compression import check_angle
from steeldesigner.core.torsion_chapter_h3 import ChapterH3, TorsionResult
from steeldesigner.core.aisc360_engine import (
    LimitStateResult, CheckBundle, Factors,
)
from math import sqrt, pi


class _AngleCompressionBundle:
    """Wrapper liviano para exponer resultado §E5 como objeto compatible con la UI."""
    def __init__(self, comp: dict):
        phi_pn_kn = comp.get("phiPn_kN") or comp.get("phiPn")
        self.phi_Pn = phi_pn_kn * 1000.0 if phi_pn_kn is not None else None  # kN → N
        self.ratio = comp.get("ratio", 0.0)
        self.limit_states = []  # tabla de verificaciones vacía para UI genérica
        self._raw = comp

    def __repr__(self):
        return f"<AngleCompression φPn={self.phi_Pn} ratio={self.ratio}>"


@dataclass
class DesignInputs:
    """Parámetros de diseño pasados al facade."""
    # Material
    Fy: float = 250.0       # MPa
    Fu: float = 400.0       # MPa
    E: float = 200_000.0    # MPa
    G: float = 77_200.0     # MPa

    # Longitudes [mm]
    Lx: float = 3000.0
    Ly: float = 3000.0
    Lb: float = 3000.0
    Kx: float = 1.0
    Ky: float = 1.0

    # Demandas
    Pu: float = 0.0    # N
    Tu_axial: float = 0.0  # N  (tracción axial)
    Mux: float = 0.0   # N·mm
    Muy: float = 0.0   # N·mm
    Vux: float = 0.0   # N
    Tu_torsion: float = 0.0  # N·mm (torsor)
    L_torsion: float = 0.0   # mm (longitud para cálculo de torsión)

    # Flexión
    Cb: float = 1.0
    stem_in_tension: bool = True

    # Cortante
    stiffeners: bool = False
    tension_field: bool = False

    # Conexión (tracción efectiva)
    U: float = 1.0
    holes_area: float = 0.0

    # Block shear
    block_shear: Optional[dict] = None

    # Ángulo: pierna conectada
    conn_leg: str = "long"    # "long" | "short"

    # Método y rigor
    method: str = "LRFD"       # "LRFD" | "ASD"
    engine_mode: str = "best_effort"  # "best_effort" | "strict"


@dataclass
class DesignResult:
    """Resultado completo del diseño."""
    section_name: str
    family_type: str
    method: str
    # Resultados por capítulo (dict de CheckBundle o None)
    tension: Any = None
    compression: Any = None
    flexure_major: Any = None
    flexure_minor: Any = None
    shear: Any = None
    interaction_ratio: float = 0.0
    passes_interaction: bool = True
    # Torsión (solo si Tu_torsion > 0)
    torsion: Optional[TorsionResult] = None
    # Propiedades geométricas resueltas
    geometry_source: dict = field(default_factory=dict)
    # Auditoría
    audit_report: Any = None
    # Pasos de cálculo (ángulos §E5)
    calc_steps: list = field(default_factory=list)
    # Errores
    error: Optional[str] = None


class EngineFacade:
    """Router principal: recibe una Section + DesignInputs y devuelve DesignResult."""

    def run(self, section: Section, inputs: DesignInputs) -> DesignResult:
        ftype = family_type(section)
        name = section.designation_modern or section.designation_legacy or "?"

        try:
            if ftype == "i_shape":
                return self._run_i_shape(section, inputs, name)
            elif ftype == "channel":
                return self._run_channel(section, inputs, name)
            elif ftype == "tee":
                return self._run_tee(section, inputs, name)
            elif ftype == "angle":
                return self._run_angle(section, inputs, name)
            elif ftype == "hss_rect":
                return self._run_hss_rect(section, inputs, name)
            elif ftype == "hss_circ":
                return self._run_hss_circ(section, inputs, name)
            else:
                return DesignResult(
                    section_name=name, family_type=ftype, method=inputs.method,
                    error=f"Familia '{section.family.family_code if section.family else '?'}' no soportada aún."
                )
        except Exception as exc:
            return DesignResult(
                section_name=name, family_type=ftype, method=inputs.method,
                error=str(exc)
            )

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _material(self, inp: DesignInputs) -> EngineMaterial:
        return EngineMaterial(Fy=inp.Fy, Fu=inp.Fu, E=inp.E, G=inp.G)

    def _lengths(self, inp: DesignInputs) -> MemberLengths:
        return MemberLengths(
            Lx=inp.Lx, Ly=inp.Ly, Lb=inp.Lb, Kx=inp.Kx, Ky=inp.Ky
        )

    def _demand(self, inp: DesignInputs) -> MemberDemand:
        return MemberDemand(
            Pu=inp.Pu, Tu=inp.Tu_axial,
            Mux=inp.Mux, Muy=inp.Muy, Vux=inp.Vux
        )

    def _method_enum(self, inp: DesignInputs) -> DesignMethod:
        return DesignMethod.LRFD if inp.method == "LRFD" else DesignMethod.ASD

    def _strictness(self, inp: DesignInputs) -> Strictness:
        return (Strictness.STRICT if inp.engine_mode == "strict"
                else Strictness.PRACTICAL)

    def _flex_input(self, inp: DesignInputs) -> FlexureInput:
        return FlexureInput(Cb=inp.Cb, stem_in_tension=inp.stem_in_tension)

    def _shear_input(self, inp: DesignInputs) -> ShearInput:
        return ShearInput(
            stiffeners_present=inp.stiffeners,
            tension_field_action=inp.tension_field,
        )

    def _eff_area(self, inp: DesignInputs) -> EffectiveAreaInput:
        return EffectiveAreaInput(U=inp.U, holes_deduction_area=inp.holes_area)

    def _block_shear(self, inp: DesignInputs) -> Optional[BlockShearInput]:
        if inp.block_shear:
            return BlockShearInput(**inp.block_shear)
        return None

    def _add_torsion(self, result: DesignResult, section: Section,
                     inp: DesignInputs) -> None:
        """Añade resultado de torsión §H3.3 para secciones abiertas."""
        if inp.Tu_torsion == 0.0:
            return
        phi_Mn_x = (result.flexure_major.controlling.design_strength
                    if result.flexure_major and result.flexure_major.controlling else 0.0)
        phi_Vn = (result.shear.controlling.design_strength
                  if result.shear and result.shear.controlling else 0.0)
        result.torsion = ChapterH3.open_section(
            Tu=inp.Tu_torsion,
            Mux=inp.Mux, Vux=inp.Vux,
            phi_Mn_x=phi_Mn_x, phi_Vn=phi_Vn,
            Cw=_safe(section.Cw_mm6),
            J=_safe(section.J_mm4),
            Wno=_safe(getattr(section, "Wno", None)),
            Sw=_safe(getattr(section, "Sw", None)),
            d=_safe(section.d), bf=_safe(section.bf),
            tf=_safe(section.tf), tw=_safe(section.tw),
            Fy=inp.Fy, E=inp.E, G=inp.G,
            L=inp.L_torsion or inp.Lx, method=inp.method,
        )

    # ------------------------------------------------------------------
    # Runners por familia
    # ------------------------------------------------------------------

    def _run_i_shape(self, sec: Section, inp: DesignInputs, name: str) -> DesignResult:
        isec = to_isection(sec)
        engine = MasterEngineV2(mode=inp.engine_mode)
        member = IShapeMember(
            section=isec,
            material=self._material(inp),
            lengths=self._lengths(inp),
            method=self._method_enum(inp),
            strictness=self._strictness(inp),
            flexure_input=self._flex_input(inp),
            shear_input=self._shear_input(inp),
            effective_area=self._eff_area(inp),
            block_shear_input=self._block_shear(inp),
        )
        raw = engine.run_i_shape_member(member, self._demand(inp))
        result = DesignResult(
            section_name=name, family_type="i_shape", method=inp.method,
            tension=raw.get("tension"),
            compression=raw.get("compression"),
            flexure_major=raw.get("flexure_major"),
            flexure_minor=raw.get("flexure_minor"),
            shear=raw.get("shear_major"),
            interaction_ratio=raw.get("interaction_ratio", 0.0),
            passes_interaction=raw.get("passes_interaction", True),
            geometry_source=raw.get("geometry_source", {}),
            audit_report=raw.get("audit_report"),
        )
        self._add_torsion(result, sec, inp)
        return result

    def _run_channel(self, sec: Section, inp: DesignInputs, name: str) -> DesignResult:
        csec = to_channel_section(sec)
        engine = MasterEngineV2(mode=inp.engine_mode)
        member = ChannelMember(
            section=csec,
            material=self._material(inp),
            lengths=self._lengths(inp),
            method=self._method_enum(inp),
            strictness=self._strictness(inp),
            flexure_input=self._flex_input(inp),
            shear_input=self._shear_input(inp),
            effective_area=self._eff_area(inp),
            block_shear_input=self._block_shear(inp),
        )
        raw = engine.run_channel_member(member, self._demand(inp))
        result = DesignResult(
            section_name=name, family_type="channel", method=inp.method,
            tension=raw.get("tension"),
            compression=raw.get("compression"),
            flexure_major=raw.get("flexure_major") or raw.get("flexure"),
            shear=raw.get("shear_major") or raw.get("shear"),
            interaction_ratio=raw.get("interaction_ratio", 0.0),
            passes_interaction=raw.get("passes_interaction", True),
            geometry_source=raw.get("geometry_source", {}),
            audit_report=raw.get("audit_report"),
        )
        self._add_torsion(result, sec, inp)
        return result

    def _run_tee(self, sec: Section, inp: DesignInputs, name: str) -> DesignResult:
        tsec = to_tee_section(sec)
        engine = MasterEngineV2(mode=inp.engine_mode)
        member = TeeMember(
            section=tsec,
            material=self._material(inp),
            lengths=self._lengths(inp),
            method=self._method_enum(inp),
            strictness=self._strictness(inp),
            flexure_input=self._flex_input(inp),
            shear_input=self._shear_input(inp),
            effective_area=self._eff_area(inp),
            block_shear_input=self._block_shear(inp),
        )
        raw = engine.run_tee_member(member, self._demand(inp))
        result = DesignResult(
            section_name=name, family_type="tee", method=inp.method,
            tension=raw.get("tension"),
            compression=raw.get("compression"),
            flexure_major=raw.get("flexure_major") or raw.get("flexure"),
            shear=raw.get("shear_major") or raw.get("shear"),
            interaction_ratio=raw.get("interaction_ratio", 0.0),
            passes_interaction=raw.get("passes_interaction", True),
            geometry_source=raw.get("geometry_source", {}),
            audit_report=raw.get("audit_report"),
        )
        self._add_torsion(result, sec, inp)
        return result

    def _run_angle(self, sec: Section, inp: DesignInputs, name: str) -> DesignResult:
        """Ángulos: usa §E5 de angle_compression para compresión."""
        args = to_angle_args(sec)
        # Compresión §E5
        comp_result = check_angle(
            b1=args["b1"], b2=args["b2"], t=args["t"],
            L=inp.Lx * inp.Kx,
            Pu=inp.Pu / 1000.0,  # check_angle espera kN
            Fy=inp.Fy, E=inp.E,
            conn_leg=inp.conn_leg,
            method=inp.method,
        )
        # También corre el motor genérico para tracción/flexión/cortante
        asec = to_angle_section(sec)
        engine = MasterEngineV2(mode=inp.engine_mode)
        member = AngleMember(
            section=asec,
            material=self._material(inp),
            lengths=self._lengths(inp),
            method=self._method_enum(inp),
            strictness=self._strictness(inp),
            flexure_input=self._flex_input(inp),
            effective_area=self._eff_area(inp),
            block_shear_input=self._block_shear(inp),
        )
        raw = engine.run_angle_member(member, self._demand(inp))

        result = DesignResult(
            section_name=name, family_type="angle", method=inp.method,
            tension=raw.get("tension"),
            compression=_AngleCompressionBundle(comp_result),
            flexure_major=raw.get("flexure") or raw.get("flexure_major"),
            shear=raw.get("shear") or raw.get("shear_major"),
            interaction_ratio=raw.get("interaction_ratio", 0.0),
            passes_interaction=raw.get("passes_interaction", True),
            geometry_source=raw.get("geometry_source", {}),
            audit_report=raw.get("audit_report"),
            calc_steps=comp_result.get("calc_steps", []),
        )
        # Adjuntar resultados §E5 como dict en geometry_source
        result.geometry_source["angle_e5"] = {
            "phiPn_kN": comp_result.get("phiPn"),
            "ratio": comp_result.get("ratio"),
            "KLr_eff": comp_result.get("KLr_eff"),
            "KLr_design": comp_result.get("KLr_design"),
            "Fcr_MPa": comp_result.get("Fcr"),
            "mode": comp_result.get("mode"),
        }
        return result

    # ------------------------------------------------------------------
    # HSS §F7 / §F8 / §G4 / §G6 — implementación directa AISC 360-22
    # ------------------------------------------------------------------

    @staticmethod
    def _hss_rect_flexure(sec: Section, Fy: float, E: float, Mux: float,
                          method: str) -> CheckBundle:
        """AISC §F7 — Flexión en HSS rectangular."""
        p, o = Factors.phi_omega(method, "flexure")
        b = CheckBundle(name="Flexure §F7")

        D = max(_safe(sec.d), 1e-6)           # altura total mm
        B = max(_safe(sec.bf), 1e-6)
        t = max(_safe(sec.tf) or _safe(sec.tw) or _safe(getattr(sec, "t_des", None)) or _safe(getattr(sec, "t_nom", None)), 1e-6)
        Zx = max(_safe(sec.Zx_mm3), 1e-6)
        Sx = max(_safe(sec.Sx_mm3), 1e-6)
        Iy = max(_safe(sec.Iy_mm4), 1e-6)
        A = max(_safe(sec.area_mm2), 1e-6)
        ry = max(_safe(sec.ry_mm) or sqrt(Iy / A), 1e-6)

        b_eff = B - 3.0 * t   # clear width of compression flange
        h_tw = (D - 2 * t) / t

        Mp = Fy * Zx

        # Classify flanges (§F7.2 Table B4.1b, case 17)
        lambda_f = b_eff / t
        lambda_pf = 1.12 * sqrt(E / Fy)
        lambda_rf = 1.40 * sqrt(E / Fy)

        # Classify web (§F7.3 Table B4.1b, case 19)
        lambda_w = h_tw
        lambda_pw = 2.42 * sqrt(E / Fy)
        lambda_rw = 5.70 * sqrt(E / Fy)

        Mn = Mp  # start with plastic moment

        # §F7.2 Flange local buckling
        if lambda_f > lambda_pf and lambda_f <= lambda_rf:
            Mn = min(Mn, Mp - (Mp - Fy * Sx) * (3.57 * lambda_f * sqrt(Fy / E) - 4.0))
        elif lambda_f > lambda_rf:
            be = min(B, 1.92 * t * sqrt(E / Fy) * (1.0 - 0.38 / lambda_f * sqrt(E / Fy)))
            Seff = Sx * (be / B) if B > 0 else Sx
            Mn = min(Mn, Fy * Seff)

        # §F7.3 Web local buckling
        if lambda_w > lambda_pw and lambda_w <= lambda_rw:
            Mn = min(Mn, Mp - (Mp - Fy * Sx) * (0.305 * lambda_w * sqrt(Fy / E) - 0.738))
        elif lambda_w > lambda_rw:
            Mn = min(Mn, Fy * Sx)

        Mn = max(Mn, 0.0)
        s = p * Mn / o
        b.results.append(LimitStateResult(
            "F", "F7", "Rect HSS flexure §F7", Mn, s, p, o, Mux,
            abs(Mux) / s if s > 0 else 0.0,
            {"b_eff/t": lambda_f, "lambda_pf": lambda_pf, "lambda_rf": lambda_rf,
             "h/t": lambda_w, "Mp_Nmm": Mp},
        ))
        return b

    @staticmethod
    def _hss_rect_shear(sec: Section, Fy: float, E: float, Vux: float,
                        method: str) -> CheckBundle:
        """AISC §G4 — Cortante en HSS rectangular (2 almas verticales)."""
        p, o = Factors.phi_omega(method, "shear")
        b = CheckBundle(name="Shear §G4")

        D = max(_safe(sec.d), 1e-6)
        t = max(_safe(sec.tf) or _safe(sec.tw) or _safe(getattr(sec, "t_des", None)) or _safe(getattr(sec, "t_nom", None)), 1e-6)
        h = D - 3.0 * t   # clear height of webs (§G4 uses h=clear height)
        h_t = h / t
        Aw = 2.0 * h * t   # two webs

        kv = 5.0   # §G4 — no transverse stiffeners
        limit_1 = 1.10 * sqrt(kv * E / Fy)
        limit_2 = 1.37 * sqrt(kv * E / Fy)
        if h_t <= limit_1:
            Cv2 = 1.0
        elif h_t <= limit_2:
            Cv2 = limit_1 / h_t
        else:
            Cv2 = 1.51 * kv * E / (h_t ** 2 * Fy)

        Vn = 0.6 * Fy * Aw * Cv2
        s = p * Vn / o
        b.results.append(LimitStateResult(
            "G", "G4", "Rect HSS shear §G4", Vn, s, p, o, Vux,
            abs(Vux) / s if s > 0 else 0.0,
            {"Aw_mm2": Aw, "Cv2": Cv2, "h/t": h_t, "kv": kv},
        ))
        return b

    @staticmethod
    def _hss_circ_flexure(sec: Section, Fy: float, E: float, Mux: float,
                          method: str) -> CheckBundle:
        """AISC §F8 — Flexión en HSS circular (tubo redondo / pipe)."""
        p, o = Factors.phi_omega(method, "flexure")
        b = CheckBundle(name="Flexure §F8")

        D = max(_safe(sec.d), 1e-6)
        t = max(_safe(sec.tf) or _safe(sec.tw) or _safe(getattr(sec, "t_des", None)) or _safe(getattr(sec, "t_nom", None)), 1e-6)
        Zx = max(_safe(sec.Zx_mm3), 1e-6)
        Sx = max(_safe(sec.Sx_mm3), 1e-6)
        D_t = D / t

        # §F8.2 limits (Table B4.1b case 20)
        lambda_p = 0.07 * E / Fy
        lambda_r = 0.31 * E / Fy

        if D_t <= lambda_p:
            Mn = Fy * Zx        # compact — plastic
            eq = "F8-1"
        elif D_t <= lambda_r:
            Mn = (0.021 * E / D_t + Fy) * Sx   # noncompact
            eq = "F8-2"
        else:
            Fcr = 0.33 * E / D_t               # slender
            Mn = Fcr * Sx
            eq = "F8-3"

        Mn = min(Mn, Fy * Zx)  # cap at Mp
        s = p * Mn / o
        b.results.append(LimitStateResult(
            "F", eq, "Round HSS/pipe flexure §F8", Mn, s, p, o, Mux,
            abs(Mux) / s if s > 0 else 0.0,
            {"D/t": D_t, "lambda_p": lambda_p, "lambda_r": lambda_r},
        ))
        return b

    @staticmethod
    def _hss_circ_shear(sec: Section, Fy: float, E: float, Vux: float,
                        method: str) -> CheckBundle:
        """AISC §G6 — Cortante en HSS circular."""
        p, o = Factors.phi_omega(method, "shear")
        b = CheckBundle(name="Shear §G6")

        D = max(_safe(sec.d), 1e-6)
        t = max(_safe(sec.tf) or _safe(sec.tw) or _safe(getattr(sec, "t_des", None)) or _safe(getattr(sec, "t_nom", None)), 1e-6)
        Ag = max(_safe(sec.area_mm2), 1e-6)
        D_t = D / t

        # §G6 — Fcr is greater of the two expressions, Table G6-1
        Fcr_a = 1.60 * E / (sqrt(max(_safe(getattr(sec, "Lx_mm", None)) or D, D)) * (D_t) ** (5.0 / 4.0))
        Fcr_b = 0.78 * E / (D_t ** (3.0 / 2.0))
        Fcr = max(Fcr_a, Fcr_b, 0.6 * Fy)
        Fcr = min(Fcr, 0.6 * Fy)   # §G6: Fcr ≤ 0.6Fy
        Vn = Fcr * Ag / 2.0
        s = p * Vn / o
        b.results.append(LimitStateResult(
            "G", "G6", "Round HSS shear §G6", Vn, s, p, o, Vux,
            abs(Vux) / s if s > 0 else 0.0,
            {"Fcr": Fcr, "D/t": D_t, "Ag_mm2": Ag},
        ))
        return b

    def _run_hss_rect(self, sec: Section, inp: DesignInputs, name: str) -> DesignResult:
        """HSS rectangular: §E7 compresión + §F7 flexión + §G4 cortante + §H3.1 torsión."""
        # Compresión §E7 via motor I-shape (misma formulación columna)
        isec = to_isection(sec)
        engine = MasterEngineV2(mode=inp.engine_mode)
        member = IShapeMember(
            section=isec,
            material=self._material(inp),
            lengths=self._lengths(inp),
            method=self._method_enum(inp),
            strictness=self._strictness(inp),
            flexure_input=self._flex_input(inp),
            shear_input=self._shear_input(inp),
            effective_area=self._eff_area(inp),
        )
        raw = engine.run_i_shape_member(member, self._demand(inp))

        # §F7 y §G4 reemplazan los resultados de flexión/cortante del motor I
        flexure = self._hss_rect_flexure(sec, inp.Fy, inp.E, inp.Mux, inp.method)
        shear = self._hss_rect_shear(sec, inp.Fy, inp.E, inp.Vux, inp.method)

        # Recalcular interacción §H1-1 con los valores correctos §F7/§G4
        comp = raw.get("compression")
        phi_Pc = getattr(comp, "controlling", None)
        phi_Pc = phi_Pc.design_strength if phi_Pc else 0.0
        phi_Mcx = flexure.controlling.design_strength if flexure.controlling else 0.0
        from steeldesigner.core.aisc360_engine import ChapterH
        ir = ChapterH.h1_1(inp.Pu, phi_Pc, inp.Mux, phi_Mcx) if (phi_Pc or phi_Mcx) else 0.0

        result = DesignResult(
            section_name=name, family_type="hss_rect", method=inp.method,
            tension=raw.get("tension"),
            compression=comp,
            flexure_major=flexure,
            shear=shear,
            interaction_ratio=ir,
            passes_interaction=ir <= 1.0,
            geometry_source=raw.get("geometry_source", {}),
            audit_report=raw.get("audit_report"),
        )

        if inp.Tu_torsion > 0:
            B = _safe(sec.bf)
            D = _safe(sec.d)
            t = _safe(sec.tf) or _safe(sec.tw) or _safe(getattr(sec, "t_des", None))
            result.torsion = ChapterH3.hss_rectangular(
                B=B, D=D, t=t, J=_safe(sec.J_mm4),
                Tu=inp.Tu_torsion, Fy=inp.Fy, method=inp.method,
            )
        return result

    def _run_hss_circ(self, sec: Section, inp: DesignInputs, name: str) -> DesignResult:
        """HSS circular: §E7 compresión + §F8 flexión + §G6 cortante + §H3.1 torsión."""
        isec = to_isection(sec)
        engine = MasterEngineV2(mode=inp.engine_mode)
        member = IShapeMember(
            section=isec,
            material=self._material(inp),
            lengths=self._lengths(inp),
            method=self._method_enum(inp),
            strictness=self._strictness(inp),
            flexure_input=self._flex_input(inp),
            shear_input=self._shear_input(inp),
            effective_area=self._eff_area(inp),
        )
        raw = engine.run_i_shape_member(member, self._demand(inp))

        # §F8 y §G6 reemplazan los resultados de flexión/cortante del motor I
        flexure = self._hss_circ_flexure(sec, inp.Fy, inp.E, inp.Mux, inp.method)
        shear = self._hss_circ_shear(sec, inp.Fy, inp.E, inp.Vux, inp.method)

        comp = raw.get("compression")
        phi_Pc = getattr(comp, "controlling", None)
        phi_Pc = phi_Pc.design_strength if phi_Pc else 0.0
        phi_Mcx = flexure.controlling.design_strength if flexure.controlling else 0.0
        from steeldesigner.core.aisc360_engine import ChapterH
        ir = ChapterH.h1_1(inp.Pu, phi_Pc, inp.Mux, phi_Mcx) if (phi_Pc or phi_Mcx) else 0.0

        result = DesignResult(
            section_name=name, family_type="hss_circ", method=inp.method,
            tension=raw.get("tension"),
            compression=comp,
            flexure_major=flexure,
            shear=shear,
            interaction_ratio=ir,
            passes_interaction=ir <= 1.0,
            geometry_source=raw.get("geometry_source", {}),
            audit_report=raw.get("audit_report"),
        )

        if inp.Tu_torsion > 0:
            D = _safe(sec.d)
            t = _safe(sec.tf) or _safe(sec.tw) or _safe(getattr(sec, "t_des", None))
            result.torsion = ChapterH3.hss_circular(
                D=D, t=t, J=_safe(sec.J_mm4),
                Tu=inp.Tu_torsion, Fy=inp.Fy, E=inp.E,
                L=inp.L_torsion or inp.Lx, method=inp.method,
            )
        return result
