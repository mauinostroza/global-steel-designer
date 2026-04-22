from __future__ import annotations
from dataclasses import dataclass
from math import sqrt
from aisc360_engine_v8_warping_integrated import ISection

@dataclass
class IShapeDims:
    d: float
    bf: float
    tf: float
    tw: float
    name: str = "Custom I"

@dataclass
class AngleDims:
    d: float
    b: float
    t: float
    name: str = "Custom L"

class SectionPropertyCalculator:
    @staticmethod
    def i_shape(dims: IShapeDims) -> ISection:
        d,bf,tf,tw = dims.d,dims.bf,dims.tf,dims.tw
        h=max(d-2*tf,0.0)
        A=2*bf*tf+h*tw
        flange_area=bf*tf
        off=d/2-tf/2
        Ix=2*(bf*tf**3/12+flange_area*off**2)+tw*h**3/12
        Iy=2*(tf*bf**3/12)+h*tw**3/12
        rx=sqrt(Ix/A) if A>0 else 0.0
        ry=sqrt(Iy/A) if A>0 else 0.0
        Sx=Ix/(d/2) if d>0 else 0.0
        Sy=Iy/(bf/2) if bf>0 else 0.0
        Zx=2*(bf*tf*(d/2-tf/2)+(tw*h/2)*(h/4))
        Zy=2*((tf*(bf/2)*(bf/4))+((h/2)*(tw/2)*(tw/4)))
        J=2*(bf*tf**3)/3+(h*tw**3)/3
        Cw=(bf**3*tf/24)*(d-tf)**2
        rts=((Iy*Cw)**0.25/(Sx**0.5)) if Iy>0 and Cw>0 and Sx>0 else 0.0
        ro=(((Ix+Iy)/A)**0.5) if A>0 else 0.0
        return ISection(name=dims.name,area=A,d=d,bf=bf,tf=tf,tw=tw,Ix=Ix,Iy=Iy,Zx=Zx,Zy=Zy,Sx=Sx,Sy=Sy,rx=rx,ry=ry,J=J,Cw=Cw,ro=ro,rts=rts)

    @staticmethod
    def angle(dims: AngleDims):
        d,b,t = dims.d,dims.b,dims.t
        A1,A2,A3 = d*t,b*t,t*t
        A=A1+A2-A3
        x1,y1 = t/2,d/2
        x2,y2 = b/2,t/2
        x3,y3 = t/2,t/2
        x_bar=(A1*x1+A2*x2-A3*x3)/A
        y_bar=(A1*y1+A2*y2-A3*y3)/A
        Ix=(t*d**3/12+A1*(y1-y_bar)**2)+(b*t**3/12+A2*(y2-y_bar)**2)-(t*t**3/12+A3*(y3-y_bar)**2)
        Iy=(d*t**3/12+A1*(x1-x_bar)**2)+(t*b**3/12+A2*(x2-x_bar)**2)-(t*t**3/12+A3*(x3-x_bar)**2)
        rx=sqrt(Ix/A) if A>0 else 0.0
        ry=sqrt(Iy/A) if A>0 else 0.0
        Sx=Ix/max(d-y_bar,y_bar)
        Sy=Iy/max(b-x_bar,x_bar)
        class AngleObj: pass
        o=AngleObj()
        o.name=dims.name; o.area=A; o.d=d; o.bf=b; o.tf=t; o.Ix=Ix; o.Iy=Iy; o.rx=rx; o.ry=ry; o.Sx=Sx; o.Sy=Sy; o.Zx=Sx; o.Zy=Sy; o.J=0.0
        return o
