from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from math import sqrt
from typing import Dict, List, Optional

class SectionClass(str, Enum):
    COMPACT = "COMPACT"
    NONCOMPACT = "NONCOMPACT"
    SLENDER = "SLENDER"

class SectionFamily(str, Enum):
    I_SHAPE = "I_SHAPE"
    CHANNEL = "CHANNEL"
    ANGLE = "ANGLE"
    TEE = "TEE"

@dataclass
class Material:
    Fy: float
    E: float = 200000.0

@dataclass
class ElementClassification:
    name: str
    element_type: str
    lambda_value: float
    lambda_p: Optional[float] = None
    lambda_r: Optional[float] = None
    section_class: Optional[SectionClass] = None
    equation_ref: str = ""
    metadata: Dict[str, float | str] = field(default_factory=dict)

@dataclass
class SectionClassificationReport:
    family: SectionFamily
    elements: List[ElementClassification] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def element(self, name: str) -> Optional[ElementClassification]:
        for item in self.elements:
            if item.name == name:
                return item
        return None

@dataclass
class IShapeInput:
    d: float
    bf: float
    tf: float
    tw: float

@dataclass
class ChannelInput:
    bf: float
    tf: float
    tw: float
    d: float

@dataclass
class AngleInput:
    d: float
    b: float
    t: float

@dataclass
class TeeInput:
    bf: float
    tf: float
    tw: float
    d: float

class B4Classifier:
    @staticmethod
    def _classify_with_p_r(name: str, element_type: str, lam: float, lam_p: float, lam_r: float, equation_ref: str) -> ElementClassification:
        if lam <= lam_p:
            cls = SectionClass.COMPACT
        elif lam <= lam_r:
            cls = SectionClass.NONCOMPACT
        else:
            cls = SectionClass.SLENDER
        return ElementClassification(name, element_type, lam, lam_p, lam_r, cls, equation_ref)

    @staticmethod
    def _classify_with_r_only(name: str, element_type: str, lam: float, lam_r: float, equation_ref: str) -> ElementClassification:
        cls = SectionClass.SLENDER if lam > lam_r else SectionClass.NONCOMPACT
        return ElementClassification(name, element_type, lam, None, lam_r, cls, equation_ref)

    @staticmethod
    def classify_i_shape_major_flexure(inp: IShapeInput, mat: Material) -> SectionClassificationReport:
        r = SectionClassificationReport(SectionFamily.I_SHAPE)
        r.elements.append(B4Classifier._classify_with_p_r("flange_major_flexure", "unstiffened", inp.bf/(2*max(inp.tf,1e-9)), 0.38*sqrt(mat.E/mat.Fy), 1.00*sqrt(mat.E/mat.Fy), "B4.1b"))
        r.elements.append(B4Classifier._classify_with_p_r("web_major_flexure", "stiffened", (inp.d-2*inp.tf)/max(inp.tw,1e-9), 3.76*sqrt(mat.E/mat.Fy), 5.70*sqrt(mat.E/mat.Fy), "B4.1b"))
        return r

    @staticmethod
    def classify_i_shape_minor_flexure(inp: IShapeInput, mat: Material) -> SectionClassificationReport:
        r = SectionClassificationReport(SectionFamily.I_SHAPE)
        r.elements.append(B4Classifier._classify_with_p_r("flange_minor_flexure", "unstiffened", inp.bf/(2*max(inp.tf,1e-9)), 0.38*sqrt(mat.E/mat.Fy), 1.00*sqrt(mat.E/mat.Fy), "B4.1b"))
        return r

    @staticmethod
    def classify_i_shape_compression(inp: IShapeInput, mat: Material) -> SectionClassificationReport:
        r = SectionClassificationReport(SectionFamily.I_SHAPE)
        r.elements.append(B4Classifier._classify_with_r_only("flange_compression", "unstiffened", inp.bf/(2*max(inp.tf,1e-9)), 0.56*sqrt(mat.E/mat.Fy), "B4.1a"))
        r.elements.append(B4Classifier._classify_with_r_only("web_compression", "stiffened", (inp.d-2*inp.tf)/max(inp.tw,1e-9), 1.49*sqrt(mat.E/mat.Fy), "B4.1a"))
        return r

    @staticmethod
    def classify_channel_major_flexure(inp: ChannelInput, mat: Material) -> SectionClassificationReport:
        r = SectionClassificationReport(SectionFamily.CHANNEL)
        r.elements.append(B4Classifier._classify_with_p_r("flange_major_flexure", "unstiffened", inp.bf/max(inp.tf,1e-9), 0.38*sqrt(mat.E/mat.Fy), 1.00*sqrt(mat.E/mat.Fy), "B4.1b"))
        r.elements.append(B4Classifier._classify_with_p_r("web_major_flexure", "stiffened", (inp.d-2*inp.tf)/max(inp.tw,1e-9), 3.76*sqrt(mat.E/mat.Fy), 5.70*sqrt(mat.E/mat.Fy), "B4.1b"))
        return r

    @staticmethod
    def classify_channel_compression(inp: ChannelInput, mat: Material) -> SectionClassificationReport:
        r = SectionClassificationReport(SectionFamily.CHANNEL)
        r.elements.append(B4Classifier._classify_with_r_only("flange_compression", "unstiffened", inp.bf/max(inp.tf,1e-9), 0.56*sqrt(mat.E/mat.Fy), "B4.1a"))
        r.elements.append(B4Classifier._classify_with_r_only("web_compression", "stiffened", (inp.d-2*inp.tf)/max(inp.tw,1e-9), 1.49*sqrt(mat.E/mat.Fy), "B4.1a"))
        return r

    @staticmethod
    def classify_angle(inp: AngleInput, mat: Material) -> SectionClassificationReport:
        r = SectionClassificationReport(SectionFamily.ANGLE)
        lam_r = 0.45*sqrt(mat.E/mat.Fy)
        r.elements.append(B4Classifier._classify_with_r_only("long_leg", "unstiffened", max(inp.d,inp.b)/max(inp.t,1e-9), lam_r, "B4"))
        r.elements.append(B4Classifier._classify_with_r_only("short_leg", "unstiffened", min(inp.d,inp.b)/max(inp.t,1e-9), lam_r, "B4"))
        return r

    @staticmethod
    def classify_tee_flexure(inp: TeeInput, mat: Material) -> SectionClassificationReport:
        r = SectionClassificationReport(SectionFamily.TEE)
        r.elements.append(B4Classifier._classify_with_p_r("flange_flexure", "unstiffened", inp.bf/(2*max(inp.tf,1e-9)), 0.38*sqrt(mat.E/mat.Fy), 1.00*sqrt(mat.E/mat.Fy), "B4.1b"))
        r.elements.append(B4Classifier._classify_with_p_r("stem_flexure", "stiffened", (inp.d-inp.tf)/max(inp.tw,1e-9), 0.84*sqrt(mat.E/mat.Fy), 1.03*sqrt(mat.E/mat.Fy), "B4.1b"))
        return r
