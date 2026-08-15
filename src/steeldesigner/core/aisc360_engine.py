from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from math import pi, sqrt
from typing import Any, Dict, List, Optional
from steeldesigner.core.aisc360_b4_classification import B4Classifier, Material as B4Material, IShapeInput, ChannelInput, AngleInput, TeeInput, SectionClass
from steeldesigner.core.section_warping import PropertyReliability, SectionWarpingShearCenter, IShapeDims as WarpIShapeDims, ChannelDims as WarpChannelDims, TeeDims as WarpTeeDims, AngleDims as WarpAngleDims

E_MPA = 200000.0
G_MPA = 77200.0

class DesignMethod(str, Enum):
    LRFD = "LRFD"
    ASD = "ASD"

class Strictness(str, Enum):
    STRICT = "STRICT"
    PRACTICAL = "PRACTICAL"

@dataclass
class Material:
    Fy: float
    Fu: float
    E: float = E_MPA
    G: float = G_MPA
    def to_b4(self): return B4Material(Fy=self.Fy, E=self.E)

@dataclass
class WarpingInputMixin:
    x_sc: Optional[float] = None
    y_sc: Optional[float] = None
    Cw: Optional[float] = None
    J: float = 0.0
    ro: Optional[float] = None

@dataclass
class ISection(WarpingInputMixin):
    name: str = ""
    area: float = 0.0
    d: float = 0.0
    bf: float = 0.0
    tf: float = 0.0
    tw: float = 0.0
    Ix: float = 0.0
    Iy: float = 0.0
    Zx: float = 0.0
    Zy: float = 0.0
    Sx: float = 0.0
    Sy: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    rts: float = 0.0
    h0: float = 0.0
    library: str = "CUSTOM"
    @property
    def ho(self): return self.h0 if self.h0 > 0 else max(self.d - self.tf, 1e-9)
    @property
    def h_web(self): return max(self.d - 2*self.tf, 1e-9)
    def b4_major(self, mat): return B4Classifier.classify_i_shape_major_flexure(IShapeInput(self.d,self.bf,self.tf,self.tw), mat.to_b4())

@dataclass
class ChannelSection(WarpingInputMixin):
    name: str = ""
    area: float = 0.0
    d: float = 0.0
    bf: float = 0.0
    tf: float = 0.0
    tw: float = 0.0
    Ix: float = 0.0
    Iy: float = 0.0
    Zx: float = 0.0
    Zy: float = 0.0
    Sx: float = 0.0
    Sy: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    rts: float = 0.0
    h0: float = 0.0
    principal_rx: float = 0.0
    principal_ry: float = 0.0
    library: str = "CUSTOM"
    @property
    def ho(self): return self.h0 if self.h0 > 0 else max(self.d - self.tf, 1e-9)
    @property
    def h_web(self): return max(self.d - 2*self.tf, 1e-9)
    def b4_major(self, mat): return B4Classifier.classify_channel_major_flexure(ChannelInput(self.bf,self.tf,self.tw,self.d), mat.to_b4())

@dataclass
class AngleSection(WarpingInputMixin):
    name: str = ""
    area: float = 0.0
    d: float = 0.0
    b: float = 0.0
    t: float = 0.0
    Ix: float = 0.0
    Iy: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    Sx: float = 0.0
    Sy: float = 0.0
    Zx: float = 0.0
    Zy: float = 0.0
    principal_rx: float = 0.0
    principal_ry: float = 0.0
    library: str = "CUSTOM"
    def b4(self, mat): return B4Classifier.classify_angle(AngleInput(self.d,self.b,self.t), mat.to_b4())

@dataclass
class TeeSection(WarpingInputMixin):
    name: str = ""
    area: float = 0.0
    d: float = 0.0
    bf: float = 0.0
    tf: float = 0.0
    tw: float = 0.0
    Ix: float = 0.0
    Iy: float = 0.0
    Sx: float = 0.0
    Sy: float = 0.0
    Zx: float = 0.0
    Zy: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    h0: float = 0.0
    principal_rx: float = 0.0
    principal_ry: float = 0.0
    library: str = "CUSTOM"
    @property
    def stem_height(self): return max(self.d - self.tf, 1e-9)
    def b4_flexure(self, mat): return B4Classifier.classify_tee_flexure(TeeInput(self.bf,self.tf,self.tw,self.d), mat.to_b4())

@dataclass
class MemberLengths:
    Lx: float = 0.0
    Ly: float = 0.0
    Lz: float = 0.0
    Lb: float = 0.0
    Kx: float = 1.0
    Ky: float = 1.0
    Kz: float = 1.0

@dataclass
class MemberDemand:
    Pu: float = 0.0
    Tu: float = 0.0
    Mux: float = 0.0
    Muy: float = 0.0
    Vux: float = 0.0
    Vuy: float = 0.0

@dataclass
class EffectiveAreaInput:
    U: float = 1.0
    An: Optional[float] = None
    holes_deduction_area: float = 0.0
    def get_areas(self, Ag):
        An = self.An if self.An is not None else max(Ag - self.holes_deduction_area, 0.0)
        return An, self.U*An

@dataclass
class BlockShearInput:
    Avg: float = 0.0
    Avn: float = 0.0
    Atg: float = 0.0
    Atn: float = 0.0
    Ubs: float = 1.0

@dataclass
class FlexureInput:
    Cb: float = 1.0
    stem_in_tension: bool = True
    connection_eccentricity_x: float = 0.0
    connection_eccentricity_y: float = 0.0

@dataclass
class ShearInput:
    a: float = 0.0
    stiffeners_present: bool = False
    tension_field_action: bool = False

@dataclass
class LimitStateResult:
    chapter: str
    equation: str
    description: str
    nominal_strength: float
    design_strength: float
    phi: float = 1.0
    omega: float = 1.0
    demand: float = 0.0
    ratio: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CheckBundle:
    name: str
    results: List[LimitStateResult] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    @property
    def controlling(self): return min(self.results, key=lambda x: x.design_strength) if self.results else None

class Factors:
    @staticmethod
    def phi_omega(method, limit_state):
        table = {"tension_yield":(0.90,1.67),"tension_rupture":(0.75,2.00),"block_shear":(0.75,2.00),"compression":(0.90,1.67),"flexure":(0.90,1.67),"shear":(0.90,1.67)}
        phi, omega = table[limit_state]
        return (phi,1.0) if method == DesignMethod.LRFD else (1.0, omega)

class GeometryResolver:
    @staticmethod
    def resolve_i(sec):
        if sec.Cw and sec.ro:
            return {"Cw":sec.Cw,"J":sec.J,"ro":sec.ro,"source":"TABULATED_OR_INPUT"}
        g=SectionWarpingShearCenter.i_shape_geometric(WarpIShapeDims(sec.d,sec.bf,sec.tf,sec.tw))
        return {"Cw":g.Cw,"J":g.J,"ro":g.ro,"source":g.warping_reliability.value}
    @staticmethod
    def resolve_channel(sec):
        if sec.Cw and sec.x_sc is not None and sec.y_sc is not None and sec.ro:
            return {"Cw":sec.Cw,"J":sec.J,"ro":sec.ro,"xo":sec.x_sc,"source":"TABULATED_OR_INPUT","reliability":PropertyReliability.TABULATED}
        g=SectionWarpingShearCenter.channel_geometric(WarpChannelDims(sec.d,sec.bf,sec.tf,sec.tw))
        xo=sec.x_sc if sec.x_sc is not None else g.x_sc
        return {"Cw":g.Cw,"J":g.J if sec.J<=0 else sec.J,"ro":g.ro,"xo":xo,"source":g.warping_reliability.value,"reliability":g.warping_reliability}
    @staticmethod
    def resolve_angle(sec):
        g=SectionWarpingShearCenter.angle_geometric(WarpAngleDims(sec.d,sec.b,sec.t))
        return {"Cw":g.Cw,"J":g.J if sec.J<=0 else sec.J,"ro":g.ro,"source":g.shear_center_reliability.value,"reliability":g.shear_center_reliability}
    @staticmethod
    def resolve_tee(sec):
        if sec.Cw and sec.x_sc is not None and sec.y_sc is not None and sec.ro and sec.principal_rx>0 and sec.principal_ry>0:
            return {"Cw":sec.Cw,"J":sec.J,"ro":sec.ro,"source":"TABULATED_OR_INPUT","reliability":PropertyReliability.TABULATED}
        g=SectionWarpingShearCenter.tee_geometric(WarpTeeDims(sec.d,sec.bf,sec.tf,sec.tw))
        return {"Cw":g.Cw,"J":g.J if sec.J<=0 else sec.J,"ro":g.ro,"source":g.warping_reliability.value,"reliability":g.warping_reliability}

class ChapterD:
    @staticmethod
    def run(area, mat, method, eff, block, demand=0.0):
        b=CheckBundle(name="Tension")
        py,oy=Factors.phi_omega(method,"tension_yield")
        Pny=mat.Fy*area
        sy=py*Pny/oy
        b.results.append(LimitStateResult("D","D2-1","Gross section yielding",Pny,sy,py,oy,demand,abs(demand)/sy if sy>0 else 0.0))
        pu,ou=Factors.phi_omega(method,"tension_rupture")
        An,Ae=eff.get_areas(area)
        Pnu=mat.Fu*Ae
        su=pu*Pnu/ou
        b.results.append(LimitStateResult("D","D2-2","Net/effective section rupture",Pnu,su,pu,ou,demand,abs(demand)/su if su>0 else 0.0,{"An":An,"Ae":Ae}))
        if block and min(block.Avg,block.Avn,block.Atg,block.Atn)>0:
            pb,ob=Factors.phi_omega(method,"block_shear")
            Rn1=0.6*mat.Fu*block.Avn+block.Ubs*mat.Fu*block.Atn
            Rn2=0.6*mat.Fy*block.Avg+block.Ubs*mat.Fu*block.Atn
            Pnb=min(Rn1,Rn2)
            sb=pb*Pnb/ob
            b.results.append(LimitStateResult("D","J4-5 / block shear","Block shear",Pnb,sb,pb,ob,demand,abs(demand)/sb if sb>0 else 0.0,{"Rn1":Rn1,"Rn2":Rn2}))
        else:
            b.notes.append("Block shear not evaluated because path areas were not supplied.")
        return b

def _eff_width_unstiffened(b, t, Fy, E, Fcr, lam_r):
    """§E7.1 Table E7.1 caso (b): elementos no rigidizados (alas, piernas). c1=0.22, c2=1.49."""
    lam = b / max(t, 1e-9)
    limit = lam_r * sqrt(Fy / max(Fcr, 1e-9))
    if lam <= limit:
        return b, False
    c1, c2 = 0.22, 1.49
    Fel = (c2 * lam_r / lam) ** 2 * Fy
    be = b * (1 - c1 * sqrt(Fel / Fcr)) * sqrt(Fel / Fcr)
    return max(min(be, b), 0.0), True

def _eff_width_stiffened(b, t, Fy, E, Fcr, lam_r):
    """§E7.1 Table E7.1 caso (a): elementos rigidizados (almas, paredes HSS). c1=0.18, c2=1.31."""
    lam = b / max(t, 1e-9)
    limit = lam_r * sqrt(Fy / max(Fcr, 1e-9))
    if lam <= limit:
        return b, False
    c1, c2 = 0.18, 1.31
    Fel = (c2 * lam_r / lam) ** 2 * Fy
    be = b * (1 - c1 * sqrt(Fel / Fcr)) * sqrt(Fel / Fcr)
    return max(min(be, b), 0.0), True

class ChapterE:
    @staticmethod
    def blocker(desc, req, demand=0.0):
        return LimitStateResult("E","BLOCKED_EXACT_DATA",desc,0.0,0.0,demand=demand,ratio=float("inf") if abs(demand)>0 else 0.0,metadata={"blocking":True,"required_inputs":req})
    @staticmethod
    def flexural_buckling(area, rx, ry, mat, lengths, method, demand=0.0, ae_fn=None):
        """§E3 pandeo flexural. Si `ae_fn(Fcr)` se entrega, aplica reducción de área
        efectiva por elementos esbeltos (§E7.1) antes de calcular Pn=Fcr·Ae."""
        p,o=Factors.phi_omega(method,"compression")
        klr=max(lengths.Kx*lengths.Lx/max(rx,1e-9), lengths.Ky*lengths.Ly/max(ry,1e-9))
        Fe=pi**2*mat.E/max(klr**2,1e-9)
        Fcr=(0.658**(mat.Fy/Fe))*mat.Fy if mat.Fy/Fe<=2.25 else 0.877*Fe
        eq="E3"; meta={"KLr":klr,"Fe":Fe}
        Ae=area
        if ae_fn is not None:
            Ae, slender_note = ae_fn(Fcr)
            if Ae < area:
                eq="E7-1"; meta["Ae"]=Ae; meta["Ag"]=area; meta.update(slender_note or {})
        Pn=Fcr*Ae
        s=p*Pn/o
        return LimitStateResult("E",eq,"Flexural buckling",Pn,s,p,o,demand,abs(demand)/s if s>0 else 0.0,meta)
    @staticmethod
    def torsional_i(sec, geom, mat, lengths, method, demand=0.0):
        if not geom.get("Cw") or not geom.get("ro") or geom.get("J",0)<=0:
            return None
        p,o=Factors.phi_omega(method,"compression")
        Lz=lengths.Lz if lengths.Lz>0 else max(lengths.Lx,lengths.Ly,1.0)
        KzL=lengths.Kz*Lz
        ro2=geom["ro"]**2
        Fe=(((pi**2)*mat.E*geom["Cw"]/max(KzL**2,1e-9))+mat.G*geom["J"])/max(sec.area*ro2,1e-9)
        Fcr=(0.658**(mat.Fy/Fe))*mat.Fy if mat.Fy/Fe<=2.25 else 0.877*Fe
        Pn=Fcr*sec.area
        s=p*Pn/o
        return LimitStateResult("E","E4","Torsional buckling, I-shape",Pn,s,p,o,demand,abs(demand)/s if s>0 else 0.0,{"source":geom.get("source")})
    @staticmethod
    def flexural_torsional_singly_symmetric(sec, geom, mat, lengths, method, demand=0.0):
        """§E4 (Eq. E4-2/E4-3) pandeo flexo-torsional para secciones monosimétricas
        (canal, tee) flectando/pandeando respecto al eje de simetría x.
        Requiere xo (distancia centro de corte-centroide) desde `geom`."""
        Cw=geom.get("Cw") or 0.0
        J=geom.get("J") or 0.0
        ro=geom.get("ro") or 0.0
        xo=geom.get("xo")
        if xo is None or ro<=0 or sec.area<=0:
            return None
        p,o=Factors.phi_omega(method,"compression")
        KLx=lengths.Kx*lengths.Lx
        rx=max(sec.rx,1e-9)
        Fex=pi**2*mat.E/max((KLx/rx)**2,1e-9)
        Lz=lengths.Lz if lengths.Lz>0 else max(lengths.Lx,lengths.Ly,1.0)
        KzL=lengths.Kz*Lz
        Fez=((pi**2*mat.E*Cw/max(KzL**2,1e-9))+mat.G*J)/max(sec.area*ro**2,1e-9) if (Cw>0 and J>0) else 0.0
        H=1.0-(xo**2)/max(ro**2,1e-9)
        if Fez<=0 or H<=0:
            return None
        disc=1.0-4.0*Fex*Fez*H/max((Fex+Fez)**2,1e-9)
        disc=max(disc,0.0)
        Fe=(Fex+Fez)/(2.0*H)*(1.0-sqrt(disc))
        Fcr=(0.658**(mat.Fy/Fe))*mat.Fy if (Fe>0 and mat.Fy/Fe<=2.25) else (0.877*Fe if Fe>0 else 0.0)
        Pn=Fcr*sec.area
        s=p*Pn/o
        return LimitStateResult("E","E4-2/E4-3","Flexural-torsional buckling (singly symmetric)",Pn,s,p,o,demand,abs(demand)/s if s>0 else 0.0,{"source":geom.get("source"),"Fex":Fex,"Fez":Fez,"H":H,"reliability":str(geom.get("reliability",""))})
    @staticmethod
    def exactness_gate_non_i(family, geom, strictness, demand=0.0):
        rel=geom.get("reliability", PropertyReliability.UNKNOWN)
        if strictness==Strictness.STRICT and rel != PropertyReliability.TABULATED:
            return ChapterE.blocker(f"Exact {family} flexo-torsional compression requires tabulated or higher-fidelity properties",["tabulated shear center","tabulated or validated Cw","principal-axis properties"],demand)
        return None

class ChapterF:
    @staticmethod
    def i_major(sec, mat, lengths, flex, method, demand=0.0):
        b4=sec.b4_major(mat)
        flange=b4.element("flange_major_flexure")
        web=b4.element("web_major_flexure")
        p,o=Factors.phi_omega(method,"flexure")
        Sx=max(sec.Sx,1e-9); Zx=max(sec.Zx,1e-9); ry=max(sec.ry,1e-9); J=max(sec.J,1e-9); ho=max(sec.ho,1e-9)
        rts=sec.rts if sec.rts>0 else max(1.1*ry,1e-9)
        Cb=max(flex.Cb,0.0); Lb=max(lengths.Lb,0.0)
        Mp=mat.Fy*Zx
        Lp=1.76*ry*sqrt(mat.E/mat.Fy)
        term=J/(Sx*ho)
        Lr=1.95*rts*(mat.E/(0.7*mat.Fy))*sqrt(term+sqrt(term**2+6.76*((0.7*mat.Fy)/mat.E)**2))
        Mr=0.7*mat.Fy*Sx
        if Lb<=Lp:
            Mn=Mp; eq="F2-1"
        elif Lb<=Lr:
            Mn=min(Mp,Cb*(Mp-(Mp-Mr)*((Lb-Lp)/max(Lr-Lp,1e-9)))); eq="F2-2"
        else:
            Fcr=(Cb*pi**2*mat.E/((Lb/rts)**2))*sqrt(1+0.078*term*(Lb/rts)**2)
            Mn=min(Fcr*Sx,Mp); eq="F2-4"
        if flange and flange.section_class==SectionClass.NONCOMPACT:
            Mn=min(Mn,Mp-(Mp-0.7*mat.Fy*Sx)*((flange.lambda_value-(flange.lambda_p or 0))/max((flange.lambda_r or 0)-(flange.lambda_p or 0),1e-9)))
        elif flange and flange.section_class==SectionClass.SLENDER:
            kc=max(0.35,min(4.0/sqrt(max(web.lambda_value if web else 1.0,1e-9)),0.76))
            Mn=min(Mn,0.9*mat.E*kc*Sx/max(flange.lambda_value**2,1e-9))
        if web and web.section_class==SectionClass.SLENDER:
            # §F5-2/F5-6: Rpg (bend-strength reduction factor), reemplaza aproximación 0.85 fija.
            hc=max(sec.h_web,1e-9)
            Afc=max(sec.bf*sec.tf,1e-9)
            aw=min((hc*sec.tw)/Afc,10.0)
            Rpg=1.0-(aw/(1200.0+300.0*aw))*(hc/max(sec.tw,1e-9)-5.7*sqrt(mat.E/mat.Fy))
            Rpg=min(max(Rpg,0.0),1.0)
            Fcr_current=Mn/Sx
            Mn=Rpg*Fcr_current*Sx
            eq="F5-2"
        s=p*Mn/o
        return LimitStateResult("F",eq,"Major-axis flexure, I-shape",Mn,s,p,o,demand,abs(demand)/s if s>0 else 0.0,{"b4":[e.__dict__ for e in b4.elements]})
    @staticmethod
    def i_minor(sec, mat, method, demand=0.0):
        """§F6 flexión eje menor con clasificación de ala (compacta/no compacta/esbelta)."""
        b4=B4Classifier.classify_i_shape_minor_flexure(IShapeInput(sec.d,sec.bf,sec.tf,sec.tw),mat.to_b4())
        flange=b4.element("flange_minor_flexure")
        p,o=Factors.phi_omega(method,"flexure")
        Sy=max(sec.Sy,1e-9); Zy=max(sec.Zy if sec.Zy>0 else sec.Sy,1e-9)
        Mp=min(mat.Fy*Zy,1.6*mat.Fy*Sy)
        eq="F6-1"; Mn=Mp
        if flange and flange.section_class==SectionClass.NONCOMPACT:
            Mn=Mp-(Mp-0.7*mat.Fy*Sy)*((flange.lambda_value-(flange.lambda_p or 0))/max((flange.lambda_r or 0)-(flange.lambda_p or 0),1e-9))
            eq="F6-2"
        elif flange and flange.section_class==SectionClass.SLENDER:
            Fcr=0.69*mat.E/max(flange.lambda_value**2,1e-9)
            Mn=Fcr*Sy
            eq="F6-3"
        Mn=max(min(Mn,Mp),0.0)
        s=p*Mn/o
        return LimitStateResult("F",eq,"Minor-axis flexure, I-shape",Mn,s,p,o,demand,abs(demand)/s if s>0 else 0.0,{"b4":[e.__dict__ for e in b4.elements]})
    @staticmethod
    def channel_major(sec, mat, lengths, flex, method, strictness, geom, demand=0.0):
        if strictness==Strictness.STRICT and geom.get("reliability") != PropertyReliability.TABULATED:
            return LimitStateResult("F","BLOCKED_MONOSYMMETRIC_FLEXURE","Channel exact flexure requires tabulated shear-center/warping data in strict mode",0.0,0.0,demand=demand,ratio=float("inf") if abs(demand)>0 else 0.0,metadata={"blocking":True})
        p,o=Factors.phi_omega(method,"flexure")
        Sx=max(sec.Sx,1e-9); Zx=max(sec.Zx,1e-9); ry=max(sec.ry,1e-9); ho=max(sec.ho,1e-9)
        J=max(geom.get("J") or 1e-9,1e-9)
        rts=sec.rts if sec.rts>0 else max(1.1*ry,1e-9)
        Cb=max(flex.Cb,0.0); Lb=max(lengths.Lb,0.0)
        Mp=mat.Fy*Zx
        # §F4: curva LTB aplicada con rts/J/Sx/ho del canal (aproximación conservadora
        # usando la forma de §F2 con propiedades monosimétricas; ver GEOMETRIC_IDEALIZED
        # en audit_report — no reemplaza la fórmula exacta F4-8 con Cb ajustado).
        Lp=1.76*ry*sqrt(mat.E/mat.Fy)
        term=J/(Sx*ho)
        Lr=1.95*rts*(mat.E/(0.7*mat.Fy))*sqrt(term+sqrt(term**2+6.76*((0.7*mat.Fy)/mat.E)**2))
        Mr=0.7*mat.Fy*Sx
        if Lb<=Lp:
            Mn=Mp; eq="F4_APPROX(F2-1)"
        elif Lb<=Lr:
            Mn=min(Mp,Cb*(Mp-(Mp-Mr)*((Lb-Lp)/max(Lr-Lp,1e-9)))); eq="F4_APPROX(F2-2)"
        else:
            Fcr=(Cb*pi**2*mat.E/((Lb/rts)**2))*sqrt(1+0.078*term*(Lb/rts)**2)
            Mn=min(Fcr*Sx,Mp); eq="F4_APPROX(F2-4)"
        s=p*Mn/o
        return LimitStateResult("F",eq,"Major-axis flexure, channel (LTB approx.)",Mn,s,p,o,demand,abs(demand)/s if s>0 else 0.0,{"source":geom.get("source"),"reliability":"GEOMETRIC_IDEALIZED","Lp":Lp,"Lr":Lr})
    @staticmethod
    def angle_flexure(sec, mat, method, strictness, geom, flex, demand_x=0.0, demand_y=0.0):
        b=CheckBundle(name="Flexure")
        if strictness==Strictness.STRICT:
            b.results.append(LimitStateResult("F","BLOCKED_ANGLE_FLEXURE","Angle exact flexure requires principal-axis/tabulated data in strict mode",0.0,0.0,demand=demand_x,ratio=float("inf") if abs(demand_x)>0 else 0.0,metadata={"blocking":True}))
            return b
        p,o=Factors.phi_omega(method,"flexure")
        Mnx=mat.Fy*max(sec.Zx if sec.Zx>0 else sec.Sx,1e-9)
        Mny=mat.Fy*max(sec.Zy if sec.Zy>0 else sec.Sy,1e-9)
        sx=p*Mnx/o; sy=p*Mny/o
        b.results.append(LimitStateResult("F","ANGLE_X","Angle flexure x-like",Mnx,sx,p,o,demand_x,abs(demand_x)/sx if sx>0 else 0.0,{"source":geom.get("source")}))
        b.results.append(LimitStateResult("F","ANGLE_Y","Angle flexure y-like",Mny,sy,p,o,demand_y,abs(demand_y)/sy if sy>0 else 0.0,{"source":geom.get("source")}))
        return b
    @staticmethod
    def tee_flexure(sec, mat, method, strictness, geom, flex, demand=0.0):
        if strictness==Strictness.STRICT and geom.get("reliability") != PropertyReliability.TABULATED:
            return LimitStateResult("F","BLOCKED_TEE_FLEXURE","Tee exact flexure requires tabulated/principal-axis data in strict mode",0.0,0.0,demand=demand,ratio=float("inf") if abs(demand)>0 else 0.0,metadata={"blocking":True})
        p,o=Factors.phi_omega(method,"flexure")
        Mn=mat.Fy*max(sec.Zx if sec.Zx>0 else sec.Sx,1e-9)
        s=p*Mn/o
        return LimitStateResult("F","TEE_B4_WARPING","Tee flexure",Mn,s,p,o,demand,abs(demand)/s if s>0 else 0.0,{"source":geom.get("source"),"stem_in_tension":flex.stem_in_tension})

class ChapterG:
    @staticmethod
    def web_shear(d_eff, tw, mat, method, demand=0.0, shear=None):
        shear=shear or ShearInput()
        p,o=Factors.phi_omega(method,"shear")
        h=max(d_eff,1e-9); tw=max(tw,1e-9)
        h_tw=h/tw; limit_1=2.24*sqrt(mat.E/mat.Fy)
        if h_tw<=limit_1:
            Cv=1.0; kv=5.34; regime="stocky web"
        else:
            if shear.a>0 and shear.stiffeners_present:
                ratio=shear.a/h; kv=5.0+5.0/max(ratio**2,1e-9) if ratio>0 else 5.34
            else:
                kv=5.34
            limit_k=1.10*sqrt(kv*mat.E/mat.Fy)
            if h_tw<=limit_k:
                Cv=limit_k/h_tw; regime="post-buckling range"
            else:
                Cv=1.51*kv*mat.E/(h_tw**2*mat.Fy); regime="elastic buckling range"
        Aw=d_eff*tw; eq="G2/G6"; notes=[]
        if shear.stiffeners_present and shear.tension_field_action and shear.a>0:
            a_h=shear.a/h
            applicable = a_h<=3.0 and a_h<=(260.0/h_tw)**2
            if applicable:
                Vn=0.6*mat.Fy*Aw*(Cv+(1.0-Cv)/(1.15*sqrt(1.0+a_h**2)))
                eq="G3-2"; regime="tension-field action"
            else:
                Vn=0.6*mat.Fy*Aw*Cv
                notes.append(f"Tension field action solicitada pero a/h={a_h:.2f} excede límites de aplicabilidad §G3.1 (a/h≤3 y a/h≤(260/(h/tw))²); se usó §G2/G6 sin campo de tensión.")
        else:
            Vn=0.6*mat.Fy*Aw*Cv
        s=p*Vn/o
        b=CheckBundle(name="Shear")
        b.results.append(LimitStateResult("G",eq,"Web shear",Vn,s,p,o,demand,abs(demand)/s if s>0 else 0.0,{"Aw":Aw,"Cv":Cv,"kv":kv,"regime":regime}))
        b.notes.extend(notes)
        return b

class ChapterH:
    @staticmethod
    def h1_1(Pu,Pc,Mux,Mcx,Muy=0.0,Mcy=1e99):
        if Pc<=0 or Mcx<=0 or Mcy<=0: return 0.0
        pr=abs(Pu)/Pc
        return pr+(8.0/9.0)*(abs(Mux)/Mcx+abs(Muy)/Mcy) if pr>=0.2 else pr/2.0+abs(Mux)/Mcx+abs(Muy)/Mcy

@dataclass
class IShapeMember:
    section:ISection
    material:Material
    lengths:MemberLengths
    method:DesignMethod=DesignMethod.LRFD
    strictness:Strictness=Strictness.STRICT
    flexure_input:FlexureInput=field(default_factory=FlexureInput)
    shear_input:ShearInput=field(default_factory=ShearInput)
    effective_area:EffectiveAreaInput=field(default_factory=EffectiveAreaInput)
    block_shear_input:Optional[BlockShearInput]=None
    def _ae_i(self, Fcr):
        sec=self.section; mat=self.material
        b4c=B4Classifier.classify_i_shape_compression(IShapeInput(sec.d,sec.bf,sec.tf,sec.tw),mat.to_b4())
        flange=b4c.element("flange_compression"); web=b4c.element("web_compression")
        Ae=sec.area; note={}
        if flange and flange.section_class==SectionClass.SLENDER:
            be,slender=_eff_width_unstiffened(sec.bf/2.0,sec.tf,mat.Fy,mat.E,Fcr,0.56*sqrt(mat.E/mat.Fy))
            if slender:
                Ae-=4.0*(sec.bf/2.0-be)*sec.tf
                note["flange_slender"]=True
        if web and web.section_class==SectionClass.SLENDER:
            be,slender=_eff_width_stiffened(sec.h_web,sec.tw,mat.Fy,mat.E,Fcr,1.49*sqrt(mat.E/mat.Fy))
            if slender:
                Ae-=(sec.h_web-be)*sec.tw
                note["web_slender"]=True
        return max(Ae,1e-6), note
    def check_all(self,d):
        g=GeometryResolver.resolve_i(self.section)
        t=ChapterD.run(self.section.area,self.material,self.method,self.effective_area,self.block_shear_input,d.Tu)
        c=CheckBundle(name="Compression")
        c.results.append(ChapterE.flexural_buckling(self.section.area,self.section.rx,self.section.ry,self.material,self.lengths,self.method,d.Pu,ae_fn=self._ae_i))
        tor=ChapterE.torsional_i(self.section,g,self.material,self.lengths,self.method,d.Pu)
        if tor: c.results.append(tor)
        fmaj=CheckBundle(name="FlexureMajor",results=[ChapterF.i_major(self.section,self.material,self.lengths,self.flexure_input,self.method,d.Mux)])
        fmin=CheckBundle(name="FlexureMinor",results=[ChapterF.i_minor(self.section,self.material,self.method,d.Muy)])
        sh=ChapterG.web_shear(self.section.h_web,self.section.tw,self.material,self.method,d.Vux,self.shear_input)
        pc=c.controlling.design_strength if c.controlling else 0.0
        mcx=fmaj.controlling.design_strength if fmaj.controlling else 0.0
        mcy=fmin.controlling.design_strength if fmin.controlling else 1e99
        inter=ChapterH.h1_1(d.Pu,pc,d.Mux,mcx,d.Muy,mcy)
        return {"tension":t,"compression":c,"flexure_major":fmaj,"flexure_minor":fmin,"shear_major":sh,"interaction_ratio":inter,"passes_interaction":inter<=1.0,"geometry_source":g}

@dataclass
class ChannelMember:
    section:ChannelSection
    material:Material
    lengths:MemberLengths
    method:DesignMethod=DesignMethod.LRFD
    strictness:Strictness=Strictness.STRICT
    flexure_input:FlexureInput=field(default_factory=FlexureInput)
    shear_input:ShearInput=field(default_factory=ShearInput)
    effective_area:EffectiveAreaInput=field(default_factory=EffectiveAreaInput)
    block_shear_input:Optional[BlockShearInput]=None
    def _ae_channel(self, Fcr):
        sec=self.section; mat=self.material
        b4c=B4Classifier.classify_channel_compression(ChannelInput(sec.bf,sec.tf,sec.tw,sec.d),mat.to_b4())
        flange=b4c.element("flange_compression"); web=b4c.element("web_compression")
        Ae=sec.area; note={}
        if flange and flange.section_class==SectionClass.SLENDER:
            be,slender=_eff_width_unstiffened(sec.bf,sec.tf,mat.Fy,mat.E,Fcr,0.56*sqrt(mat.E/mat.Fy))
            if slender:
                Ae-=2.0*(sec.bf-be)*sec.tf
                note["flange_slender"]=True
        if web and web.section_class==SectionClass.SLENDER:
            be,slender=_eff_width_stiffened(sec.h_web,sec.tw,mat.Fy,mat.E,Fcr,1.49*sqrt(mat.E/mat.Fy))
            if slender:
                Ae-=(sec.h_web-be)*sec.tw
                note["web_slender"]=True
        return max(Ae,1e-6), note
    def check_all(self,d):
        g=GeometryResolver.resolve_channel(self.section)
        t=ChapterD.run(self.section.area,self.material,self.method,self.effective_area,self.block_shear_input,d.Tu)
        c=CheckBundle(name="Compression")
        c.results.append(ChapterE.flexural_buckling(self.section.area,self.section.rx,self.section.ry,self.material,self.lengths,self.method,d.Pu,ae_fn=self._ae_channel))
        if self.strictness==Strictness.STRICT:
            b=ChapterE.exactness_gate_non_i("channel",g,self.strictness,d.Pu)
            if b: c.results.append(b)
        else:
            eft=ChapterE.flexural_torsional_singly_symmetric(self.section,g,self.material,self.lengths,self.method,d.Pu)
            if eft: c.results.append(eft)
        f=CheckBundle(name="FlexureMajor",results=[ChapterF.channel_major(self.section,self.material,self.lengths,self.flexure_input,self.method,self.strictness,g,d.Mux)])
        sh=ChapterG.web_shear(self.section.h_web,self.section.tw,self.material,self.method,d.Vux,self.shear_input)
        pc=c.controlling.design_strength if c.controlling else 0.0
        mcx=f.controlling.design_strength if f.controlling else 0.0
        inter=ChapterH.h1_1(d.Pu,pc,d.Mux,mcx)
        return {"tension":t,"compression":c,"flexure_major":f,"shear":sh,"interaction_ratio":inter,"passes_interaction":inter<=1.0,"geometry_source":g}

@dataclass
class AngleMember:
    section:AngleSection
    material:Material
    lengths:MemberLengths
    method:DesignMethod=DesignMethod.LRFD
    strictness:Strictness=Strictness.STRICT
    flexure_input:FlexureInput=field(default_factory=FlexureInput)
    effective_area:EffectiveAreaInput=field(default_factory=EffectiveAreaInput)
    block_shear_input:Optional[BlockShearInput]=None
    def check_all(self,d):
        g=GeometryResolver.resolve_angle(self.section)
        t=ChapterD.run(self.section.area,self.material,self.method,self.effective_area,self.block_shear_input,d.Tu)
        c=CheckBundle(name="Compression")
        c.results.append(ChapterE.flexural_buckling(self.section.area,self.section.rx,self.section.ry,self.material,self.lengths,self.method,d.Pu))
        b=ChapterE.exactness_gate_non_i("angle",g,self.strictness,d.Pu)
        if b: c.results.append(b)
        f=ChapterF.angle_flexure(self.section,self.material,self.method,self.strictness,g,self.flexure_input,d.Mux,d.Muy)
        phi=0.90 if self.method==DesignMethod.LRFD else 1.0
        om=1.0 if self.method==DesignMethod.LRFD else 1.67
        ds=phi*0.6*self.material.Fy*self.section.t*max(self.section.d,self.section.b)/om
        sh=CheckBundle(name="Shear",results=[LimitStateResult("G","ANGLE_SHEAR_BASE","Angle shear baseline",ds*om/max(phi,1e-9),ds,phi,om,d.Vux,abs(d.Vux)/ds if ds>0 else 0.0)])
        pc=c.controlling.design_strength if c.controlling else 0.0
        mcx=f.results[0].design_strength if f.results else 0.0
        mcy=f.results[1].design_strength if len(f.results)>1 else 1e99
        inter=ChapterH.h1_1(d.Pu,pc,d.Mux,mcx,d.Muy,mcy)
        return {"tension":t,"compression":c,"flexure":f,"shear":sh,"interaction_ratio":inter,"passes_interaction":inter<=1.0,"geometry_source":g}

@dataclass
class TeeMember:
    section:TeeSection
    material:Material
    lengths:MemberLengths
    method:DesignMethod=DesignMethod.LRFD
    strictness:Strictness=Strictness.STRICT
    flexure_input:FlexureInput=field(default_factory=FlexureInput)
    effective_area:EffectiveAreaInput=field(default_factory=EffectiveAreaInput)
    block_shear_input:Optional[BlockShearInput]=None
    def check_all(self,d):
        g=GeometryResolver.resolve_tee(self.section)
        t=ChapterD.run(self.section.area,self.material,self.method,self.effective_area,self.block_shear_input,d.Tu)
        c=CheckBundle(name="Compression")
        c.results.append(ChapterE.flexural_buckling(self.section.area,self.section.rx,self.section.ry,self.material,self.lengths,self.method,d.Pu))
        b=ChapterE.exactness_gate_non_i("tee",g,self.strictness,d.Pu)
        if b: c.results.append(b)
        f=CheckBundle(name="Flexure",results=[ChapterF.tee_flexure(self.section,self.material,self.method,self.strictness,g,self.flexure_input,d.Mux)])
        phi=0.90 if self.method==DesignMethod.LRFD else 1.0
        om=1.0 if self.method==DesignMethod.LRFD else 1.67
        ds=phi*0.6*self.material.Fy*self.section.tw*self.section.stem_height/om
        sh=CheckBundle(name="Shear",results=[LimitStateResult("G","TEE_SHEAR_BASE","Tee shear baseline",ds*om/max(phi,1e-9),ds,phi,om,d.Vux,abs(d.Vux)/ds if ds>0 else 0.0)])
        pc=c.controlling.design_strength if c.controlling else 0.0
        mcx=f.controlling.design_strength if f.controlling else 0.0
        inter=ChapterH.h1_1(d.Pu,pc,d.Mux,mcx)
        return {"tension":t,"compression":c,"flexure":f,"shear":sh,"interaction_ratio":inter,"passes_interaction":inter<=1.0,"geometry_source":g}
