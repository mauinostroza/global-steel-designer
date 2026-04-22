from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from math import sqrt
from typing import Optional, Dict

class PropertyReliability(str, Enum):
    TABULATED = "TABULATED"
    GEOMETRIC_IDEALIZED = "GEOMETRIC_IDEALIZED"
    APPROXIMATE = "APPROXIMATE"
    UNKNOWN = "UNKNOWN"

@dataclass
class WarpingShearCenterResult:
    x_sc: Optional[float] = None
    y_sc: Optional[float] = None
    Cw: Optional[float] = None
    J: Optional[float] = None
    ro: Optional[float] = None
    metadata: Dict[str, float | str | bool] | None = None
    shear_center_reliability: PropertyReliability = PropertyReliability.UNKNOWN
    warping_reliability: PropertyReliability = PropertyReliability.UNKNOWN

@dataclass
class IShapeDims:
    d: float
    bf: float
    tf: float
    tw: float

@dataclass
class ChannelDims:
    d: float
    bf: float
    tf: float
    tw: float

@dataclass
class TeeDims:
    d: float
    bf: float
    tf: float
    tw: float

@dataclass
class AngleDims:
    d: float
    b: float
    t: float

class SectionWarpingShearCenter:
    @staticmethod
    def _ro(Ix, Iy, A):
        return sqrt((Ix+Iy)/A) if A > 0 else 0.0

    @staticmethod
    def i_shape_geometric(d):
        h=max(d.d-2*d.tf,0.0)
        A=2*d.bf*d.tf+h*d.tw
        Ix=2*(d.bf*d.tf**3/12+d.bf*d.tf*(d.d/2-d.tf/2)**2)+d.tw*h**3/12
        Iy=2*(d.tf*d.bf**3/12)+h*d.tw**3/12
        J=2*(d.bf*d.tf**3)/3+(h*d.tw**3)/3
        Cw=(d.bf**3*d.tf/24)*(d.d-d.tf)**2
        return WarpingShearCenterResult(d.bf/2,d.d/2,Cw,J,SectionWarpingShearCenter._ro(Ix,Iy,A),{},PropertyReliability.GEOMETRIC_IDEALIZED,PropertyReliability.GEOMETRIC_IDEALIZED)

    @staticmethod
    def channel_geometric(d):
        h=max(d.d-2*d.tf,0.0)
        A=2*d.bf*d.tf+h*d.tw
        xw=d.tw/2; xf=d.bf/2
        xb=(h*d.tw*xw+2*d.bf*d.tf*xf)/A
        yb=d.d/2
        Ix=d.tw*h**3/12+2*(d.bf*d.tf**3/12+d.bf*d.tf*(d.d/2-d.tf/2)**2)
        Iy=h*d.tw**3/12+h*d.tw*(xw-xb)**2+2*(d.tf*d.bf**3/12+d.bf*d.tf*(xf-xb)**2)
        J=(h*d.tw**3)/3+2*(d.bf*d.tf**3)/3
        Cw=(d.bf**3*d.tf/12)*(d.d-d.tf)**2/2
        return WarpingShearCenterResult(-0.25*d.bf,yb,Cw,J,SectionWarpingShearCenter._ro(Ix,Iy,A),{},PropertyReliability.APPROXIMATE,PropertyReliability.APPROXIMATE)

    @staticmethod
    def tee_geometric(d):
        h=max(d.d-d.tf,0.0)
        A=d.bf*d.tf+h*d.tw
        xb=d.bf/2
        yb=(d.bf*d.tf*(d.d-d.tf/2)+h*d.tw*(h/2))/A
        Ix=d.bf*d.tf**3/12+d.bf*d.tf*(d.d-d.tf/2-yb)**2+d.tw*h**3/12+d.tw*h*(h/2-yb)**2
        Iy=d.tf*d.bf**3/12+h*d.tw**3/12
        J=(d.bf*d.tf**3)/3+(h*d.tw**3)/3
        Cw=(d.bf**3*d.tf/12)*(d.d-d.tf/2-yb)**2
        return WarpingShearCenterResult(xb,yb,Cw,J,SectionWarpingShearCenter._ro(Ix,Iy,A),{},PropertyReliability.APPROXIMATE,PropertyReliability.APPROXIMATE)

    @staticmethod
    def angle_geometric(d):
        A1,A2,A3=d.d*d.t,d.b*d.t,d.t*d.t
        A=A1+A2-A3
        x1,y1=d.t/2,d.d/2
        x2,y2=d.b/2,d.t/2
        x3,y3=d.t/2,d.t/2
        xb=(A1*x1+A2*x2-A3*x3)/A
        yb=(A1*y1+A2*y2-A3*y3)/A
        Ix=(d.t*d.d**3/12+A1*(y1-yb)**2)+(d.b*d.t**3/12+A2*(y2-yb)**2)-(d.t*d.t**3/12+A3*(y3-yb)**2)
        Iy=(d.d*d.t**3/12+A1*(x1-xb)**2)+(d.t*d.b**3/12+A2*(x2-xb)**2)-(d.t*d.t**3/12+A3*(x3-xb)**2)
        J=((d.d-d.t/2)*d.t**3)/3+((d.b-d.t/2)*d.t**3)/3
        return WarpingShearCenterResult(xb-0.35*d.b,yb-0.35*d.d,None,J,SectionWarpingShearCenter._ro(Ix,Iy,A),{},PropertyReliability.APPROXIMATE,PropertyReliability.UNKNOWN)
