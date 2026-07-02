"""
Drawer — generación de primitivas de dibujo para perfiles estructurales.

Esta capa NO depende de PyQt. Produce una lista de primitivas
(Line, Arc, Dimension, Text) que luego la UI renderiza en QGraphicsScene,
o que se pueden exportar a SVG/PNG.

8 plantillas soportadas:
    I_welded       — IN, HN, IP, IE, PH, HR, T (soldados chilenos)
    I_rolled       — W, HP, WT, M, S, IPE, IPN, HEA, HEB, HEM, HL, HD
    channel_rolled — C, MC, UPN, UPE (canales laminados)
    channel_cf     — C, CA, IC, ICA, OC, OCA (canales conformados en frío)
    angle          — L (laminada y plegada), 2L
    hss_rect       — HSS rectangular, Cajón, CJ, CJE
    hss_circ       — HSS circular, O (tubos)
    tee            — WT, MT, ST, T

Convención de coordenadas:
    Origen en el centroide de la sección.
    Y hacia arriba (positivo), X hacia la derecha (positivo).
    Todas las unidades en mm.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .models import Section


# ----------------------------------------------------------------------
# Primitivas
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Primitive:
    """Primitiva base de dibujo."""
    type: str  # 'line', 'arc', 'dimension', 'text', 'circle'
    layer: str = "outline"  # 'outline', 'dimension', 'centerline', 'fill'


@dataclass(frozen=True, slots=True)
class Line(Primitive):
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    type: str = "line"


@dataclass(frozen=True, slots=True)
class Arc(Primitive):
    cx: float = 0.0
    cy: float = 0.0
    radius: float = 0.0
    start_angle_deg: float = 0.0
    span_angle_deg: float = 90.0
    type: str = "arc"


@dataclass(frozen=True, slots=True)
class Circle(Primitive):
    cx: float = 0.0
    cy: float = 0.0
    radius: float = 0.0
    type: str = "circle"


@dataclass(frozen=True, slots=True)
class Dimension(Primitive):
    """Cota (línea de extensión + línea de cota + texto)."""
    kind: str = "horizontal"  # 'horizontal', 'vertical'
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    text: str = ""
    type: str = "dimension"
    layer: str = "dimension"


@dataclass(frozen=True, slots=True)
class Text(Primitive):
    x: float = 0.0
    y: float = 0.0
    text: str = ""
    size: float = 12.0
    type: str = "text"
    layer: str = "text"


# ----------------------------------------------------------------------
# API principal
# ----------------------------------------------------------------------
def draw_section(section: Section) -> List[Primitive]:
    """Genera la lista de primitivas para dibujar un perfil.

    Selecciona la plantilla según `section.family.drawing_template`.
    """
    template = section.family.drawing_template
    if template == "I_welded":
        return _draw_i_welded(section)
    if template == "I_rolled":
        return _draw_i_rolled(section)
    if template == "channel_rolled":
        return _draw_channel_rolled(section)
    if template == "channel_cf":
        return _draw_channel_cf(section)
    if template == "angle":
        return _draw_angle(section)
    if template == "hss_rect":
        return _draw_hss_rect(section)
    if template == "hss_circ":
        return _draw_hss_circ(section)
    if template == "tee":
        return _draw_tee(section)
    # Fallback: dibujar un rectángulo simple
    return _draw_fallback(section)


# ----------------------------------------------------------------------
# Helper para dimensiones automáticas
# ----------------------------------------------------------------------
def _add_horizontal_dim(primitives: List[Primitive], x1: float, x2: float,
                        y: float, text: str, offset: float = 20.0) -> None:
    """Agrega una cota horizontal por encima (y+offset)."""
    primitives.append(Dimension(
        kind="horizontal",
        x1=x1, y1=y, x2=x2, y2=y,
        text=text,
        # La UI interpretará y+offset como posición de la línea de cota
    ))


def _add_vertical_dim(primitives: List[Primitive], y1: float, y2: float,
                      x: float, text: str, offset: float = 20.0) -> None:
    """Agrega una cota vertical a la derecha (x+offset)."""
    primitives.append(Dimension(
        kind="vertical",
        x1=x, y1=y1, x2=x, y2=y2,
        text=text,
    ))


def _fmt(value: Optional[float], decimals: int = 1) -> str:
    if value is None:
        return "—"
    if value == int(value):
        return str(int(value))
    return f"{value:.{decimals}f}"


# ----------------------------------------------------------------------
# Plantilla: I_welded (IN, HN, IP, IE, PH, HR, T)
# ----------------------------------------------------------------------
def _draw_i_welded(section: Section) -> List[Primitive]:
    """Dibuja un perfil I soldado (sin radios, soldadura como línea gruesa).

    Geometría: alma + 2 alas rectangulares. La soldadura se representa
    como una línea gruesa en la intersección alma-ala.
    """
    d = section.d or 0.0
    bf = section.bf or 0.0
    tf = section.tf or 0.0
    tw = section.tw or 0.0

    if d == 0 or bf == 0:
        return _draw_fallback(section)

    prims: List[Primitive] = []

    # Coordenadas clave
    y_top = d / 2
    y_bot = -d / 2
    y_top_in = y_top - tf
    y_bot_in = y_bot + tf
    x_left = -bf / 2
    x_right = bf / 2
    x_left_web = -tw / 2
    x_right_web = tw / 2

    # Ala superior (rectángulo)
    prims.append(Line(x1=x_left, y1=y_top, x2=x_right, y2=y_top))
    prims.append(Line(x1=x_right, y1=y_top, x2=x_right, y2=y_top_in))
    prims.append(Line(x1=x_right, y1=y_top_in, x2=x_right_web, y2=y_top_in))
    prims.append(Line(x1=x_left_web, y1=y_top_in, x2=x_left, y2=y_top_in))
    prims.append(Line(x1=x_left, y1=y_top_in, x2=x_left, y2=y_top))

    # Alma (líneas verticales)
    prims.append(Line(x1=x_right_web, y1=y_top_in, x2=x_right_web, y2=y_bot_in))
    prims.append(Line(x1=x_left_web, y1=y_top_in, x2=x_left_web, y2=y_bot_in))

    # Ala inferior (rectángulo)
    prims.append(Line(x1=x_left, y1=y_bot, x2=x_right, y2=y_bot))
    prims.append(Line(x1=x_right, y1=y_bot, x2=x_right, y2=y_bot_in))
    prims.append(Line(x1=x_right, y1=y_bot_in, x2=x_right_web, y2=y_bot_in))
    prims.append(Line(x1=x_left_web, y1=y_bot_in, x2=x_left, y2=y_bot_in))
    prims.append(Line(x1=x_left, y1=y_bot_in, x2=x_left, y2=y_bot))

    # Soldadura (líneas gruesas en intersecciones)
    prims.append(Line(x1=x_left_web, y1=y_top_in, x2=x_right_web, y2=y_top_in,
                       layer="fill"))
    prims.append(Line(x1=x_left_web, y1=y_bot_in, x2=x_right_web, y2=y_bot_in,
                       layer="fill"))

    # Cotas
    _add_vertical_dim(prims, y_bot, y_top, x_right + 20, f"d = {_fmt(d)}")
    _add_horizontal_dim(prims, x_left, x_right, y_top + 20, f"bf = {_fmt(bf)}")
    prims.append(Dimension(kind="horizontal", x1=-tw/2, y1=0, x2=tw/2, y2=0,
                            text=f"tw = {_fmt(tw)}"))
    prims.append(Dimension(kind="horizontal", x1=x_right - tf, y1=y_top,
                            x2=x_right, y2=y_top, text=f"tf = {_fmt(tf)}"))

    # Centroide
    prims.append(Line(x1=-10, y1=0, x2=10, y2=0, layer="centerline"))
    prims.append(Line(x1=0, y1=-10, x2=0, y2=10, layer="centerline"))

    return prims


# ----------------------------------------------------------------------
# Plantilla: I_rolled (W, HP, WT, IPE, IPN, HEA, HEB, HEM, HL, HD)
# ----------------------------------------------------------------------
def _draw_i_rolled(section: Section) -> List[Primitive]:
    """Dibuja un perfil I laminado con radios de empalme r."""
    d = section.d or 0.0
    bf = section.bf or 0.0
    tf = section.tf or 0.0
    tw = section.tw or 0.0
    r = section.r or 0.0

    if d == 0 or bf == 0:
        return _draw_fallback(section)

    prims: List[Primitive] = []

    y_top = d / 2
    y_bot = -d / 2
    y_top_in = y_top - tf
    y_bot_in = y_bot + tf
    x_left = -bf / 2
    x_right = bf / 2
    x_left_web = -tw / 2
    x_right_web = tw / 2

    # Ala superior
    prims.append(Line(x1=x_left, y1=y_top, x2=x_right, y2=y_top))
    prims.append(Line(x1=x_right, y1=y_top, x2=x_right, y2=y_top_in))
    prims.append(Line(x1=x_right, y1=y_top_in, x2=x_right_web + r, y2=y_top_in))

    # Radio superior derecho
    if r > 0:
        prims.append(Arc(cx=x_right_web + r, cy=y_top_in - r, radius=r,
                          start_angle_deg=90, span_angle_deg=-90))
        # Radio superior izquierdo
        prims.append(Arc(cx=x_left_web - r, cy=y_top_in - r, radius=r,
                          start_angle_deg=0, span_angle_deg=-90))
    prims.append(Line(x1=x_left_web - r, y1=y_top_in, x2=x_left, y2=y_top_in))
    prims.append(Line(x1=x_left, y1=y_top_in, x2=x_left, y2=y_top))

    # Alma
    prims.append(Line(x1=x_right_web, y1=y_top_in - r, x2=x_right_web, y2=y_bot_in + r))
    prims.append(Line(x1=x_left_web, y1=y_top_in - r, x2=x_left_web, y2=y_bot_in + r))

    # Ala inferior
    prims.append(Line(x1=x_left, y1=y_bot, x2=x_right, y2=y_bot))
    prims.append(Line(x1=x_right, y1=y_bot, x2=x_right, y2=y_bot_in))
    prims.append(Line(x1=x_right, y1=y_bot_in, x2=x_right_web + r, y2=y_bot_in))
    if r > 0:
        prims.append(Arc(cx=x_right_web + r, cy=y_bot_in + r, radius=r,
                          start_angle_deg=0, span_angle_deg=90))
        prims.append(Arc(cx=x_left_web - r, cy=y_bot_in + r, radius=r,
                          start_angle_deg=180, span_angle_deg=90))
    prims.append(Line(x1=x_left_web - r, y1=y_bot_in, x2=x_left, y2=y_bot_in))
    prims.append(Line(x1=x_left, y1=y_bot_in, x2=x_left, y2=y_bot))

    # Cotas
    _add_vertical_dim(prims, y_bot, y_top, x_right + 20, f"d = {_fmt(d)}")
    _add_horizontal_dim(prims, x_left, x_right, y_top + 20, f"bf = {_fmt(bf)}")
    prims.append(Dimension(kind="horizontal", x1=-tw/2, y1=0, x2=tw/2, y2=0,
                            text=f"tw = {_fmt(tw)}"))
    prims.append(Dimension(kind="horizontal", x1=x_right - tf, y1=y_top,
                            x2=x_right, y2=y_top, text=f"tf = {_fmt(tf)}"))

    # Centroide
    prims.append(Line(x1=-10, y1=0, x2=10, y2=0, layer="centerline"))
    prims.append(Line(x1=0, y1=-10, x2=0, y2=10, layer="centerline"))

    return prims


# ----------------------------------------------------------------------
# Plantilla: channel_rolled (C, MC, UPN, UPE)
# ----------------------------------------------------------------------
def _draw_channel_rolled(section: Section) -> List[Primitive]:
    """Dibuja un canal laminado (sección C con alas asimétricas)."""
    d = section.d or 0.0
    bf = section.bf or 0.0
    tf = section.tf or 0.0
    tw = section.tw or 0.0
    r = section.r or 0.0

    if d == 0 or bf == 0:
        return _draw_fallback(section)

    prims: List[Primitive] = []

    # Canal: alma en X=0, alas hacia la derecha (X positivo)
    # El centroide está desplazado hacia el alma
    y_top = d / 2
    y_bot = -d / 2
    y_top_in = y_top - tf
    y_bot_in = y_bot + tf
    x_back = 0.0  # cara del alma
    x_front = bf  # cara exterior del ala

    # Alma (vertical, a la izquierda)
    prims.append(Line(x1=x_back, y1=y_top, x2=x_back, y2=y_bot))
    prims.append(Line(x1=x_back - tw, y1=y_top, x2=x_back - tw, y2=y_bot))

    # Ala superior
    prims.append(Line(x1=x_back - tw, y1=y_top, x2=x_front, y2=y_top))
    prims.append(Line(x1=x_front, y1=y_top, x2=x_front, y2=y_top_in))
    prims.append(Line(x1=x_front, y1=y_top_in, x2=x_back + r, y2=y_top_in))
    if r > 0:
        prims.append(Arc(cx=x_back + r, cy=y_top_in - r, radius=r,
                          start_angle_deg=90, span_angle_deg=-90))

    # Ala inferior
    prims.append(Line(x1=x_back - tw, y1=y_bot, x2=x_front, y2=y_bot))
    prims.append(Line(x1=x_front, y1=y_bot, x2=x_front, y2=y_bot_in))
    prims.append(Line(x1=x_front, y1=y_bot_in, x2=x_back + r, y2=y_bot_in))
    if r > 0:
        prims.append(Arc(cx=x_back + r, cy=y_bot_in + r, radius=r,
                          start_angle_deg=0, span_angle_deg=90))

    # Cotas
    _add_vertical_dim(prims, y_bot, y_top, x_front + 20, f"d = {_fmt(d)}")
    _add_horizontal_dim(prims, x_back - tw, x_front, y_top + 20, f"bf = {_fmt(bf)}")
    prims.append(Dimension(kind="horizontal", x1=x_back - tw, y1=0, x2=x_back, y2=0,
                            text=f"tw = {_fmt(tw)}"))
    prims.append(Dimension(kind="horizontal", x1=x_front - tf, y1=y_top,
                            x2=x_front, y2=y_top, text=f"tf = {_fmt(tf)}"))

    return prims


# ----------------------------------------------------------------------
# Plantilla: channel_cf (C, CA, IC, ICA, OC, OCA conformados en frío)
# ----------------------------------------------------------------------
def _draw_channel_cf(section: Section) -> List[Primitive]:
    """Dibuja un canal conformado en frío con radio R en pliegues.

    CA tiene ala atiesada (C_dim). IC/ICA son dobles back-to-back.
    """
    d = section.d or 0.0
    bf = section.bf or 0.0
    t = section.t_nom or 0.0
    R = section.R_ext or 0.0
    C_dim = section.C_dim or 0.0

    if d == 0 or bf == 0 or t == 0:
        return _draw_fallback(section)

    prims: List[Primitive] = []

    y_top = d / 2
    y_bot = -d / 2
    x_back = -t / 2  # centro del alma
    x_front = bf - t / 2

    # Alma (línea central en x_back)
    prims.append(Line(x1=x_back - t/2, y1=y_top - R, x2=x_back - t/2, y2=y_bot + R))
    prims.append(Line(x1=x_back + t/2, y1=y_top - R, x2=x_back + t/2, y2=y_bot + R))

    # Ala superior
    prims.append(Line(x1=x_back - t/2, y1=y_top, x2=x_front, y2=y_top))
    prims.append(Line(x1=x_front, y1=y_top, x2=x_front, y2=y_top - t))
    prims.append(Line(x1=x_front, y1=y_top - t, x2=x_back + t/2 + R, y2=y_top - t))

    # Pliegue superior (radio R)
    if R > 0:
        prims.append(Arc(cx=x_back + t/2 + R, cy=y_top - R, radius=R,
                          start_angle_deg=90, span_angle_deg=-90))

    # Ala inferior
    prims.append(Line(x1=x_back - t/2, y1=y_bot, x2=x_front, y2=y_bot))
    prims.append(Line(x1=x_front, y1=y_bot, x2=x_front, y2=y_bot + t))
    prims.append(Line(x1=x_front, y1=y_bot + t, x2=x_back + t/2 + R, y2=y_bot + t))
    if R > 0:
        prims.append(Arc(cx=x_back + t/2 + R, cy=y_bot + R, radius=R,
                          start_angle_deg=0, span_angle_deg=90))

    # Atiesador (CA, ICA): línea vertical en el extremo del ala
    if C_dim > 0:
        prims.append(Line(x1=x_front, y1=y_top, x2=x_front, y2=y_top - C_dim))
        prims.append(Line(x1=x_front, y1=y_bot, x2=x_front, y2=y_bot + C_dim))

    # Cotas
    _add_vertical_dim(prims, y_bot, y_top, x_front + 30, f"d = {_fmt(d)}")
    _add_horizontal_dim(prims, x_back - t/2, x_front, y_top + 20, f"bf = {_fmt(bf)}")
    prims.append(Dimension(kind="horizontal", x1=x_back - t/2, y1=0, x2=x_back + t/2, y2=0,
                            text=f"t = {_fmt(t)}"))
    if C_dim > 0:
        prims.append(Dimension(kind="vertical", x1=x_front + 5, y1=y_top,
                                x2=x_front + 5, y2=y_top - C_dim,
                                text=f"C = {_fmt(C_dim)}"))

    return prims


# ----------------------------------------------------------------------
# Plantilla: angle (L laminada y plegada, 2L)
# ----------------------------------------------------------------------
def _draw_angle(section: Section) -> List[Primitive]:
    """Dibuja un ángulo L (laminada o plegada)."""
    d = section.d or 0.0
    bf = section.bf or d
    t = section.tf or section.t_nom or 0.0
    r = section.r or 0.0
    R = section.R_ext or 0.0

    if d == 0 or t == 0:
        return _draw_fallback(section)

    prims: List[Primitive] = []

    # L: ala horizontal + alma vertical, encontrándose en esquina inferior-izquierda
    # Coordenadas con origen en la esquina exterior inferior-izquierda
    x_left = 0.0
    x_right = bf
    y_top = d
    y_bot = 0.0

    # Ala horizontal (en la parte superior)
    prims.append(Line(x1=x_left, y1=y_top, x2=x_right, y2=y_top))
    prims.append(Line(x1=x_right, y1=y_top, x2=x_right, y2=y_top - t))
    prims.append(Line(x1=x_right, y1=y_top - t, x2=x_left + t + (r if r > 0 else R), y2=y_top - t))

    # Alma vertical (en la parte izquierda)
    prims.append(Line(x1=x_left, y1=y_top, x2=x_left, y2=y_bot))
    prims.append(Line(x1=x_left, y1=y_bot, x2=x_left + t, y2=y_bot))
    prims.append(Line(x1=x_left + t, y1=y_bot, x2=x_left + t, y2=y_top - t - (r if r > 0 else R)))

    # Radio en la intersección
    if r > 0:  # L laminada
        prims.append(Arc(cx=x_left + t + r, cy=y_top - t - r, radius=r,
                          start_angle_deg=180, span_angle_deg=90))
    elif R > 0:  # L plegada (radio exterior)
        prims.append(Arc(cx=x_left + t + R, cy=y_top - t - R, radius=R,
                          start_angle_deg=180, span_angle_deg=90))

    # Cotas
    _add_vertical_dim(prims, y_bot, y_top, x_right + 20, f"d = {_fmt(d)}")
    _add_horizontal_dim(prims, x_left, x_right, y_top + 20, f"bf = {_fmt(bf)}")
    prims.append(Dimension(kind="horizontal", x1=x_left, y1=y_bot - 10, x2=x_left + t, y2=y_bot - 10,
                            text=f"t = {_fmt(t)}"))

    return prims


# ----------------------------------------------------------------------
# Plantilla: hss_rect (HSS rectangular, Cajón, CJ, CJE)
# ----------------------------------------------------------------------
def _draw_hss_rect(section: Section) -> List[Primitive]:
    """Dibuja un cajón rectangular hueco con esquinas redondeadas R."""
    d = section.d or 0.0
    bf = section.bf or section.B or 0.0
    t = section.t_nom or section.tf or 0.0
    R = section.R_ext or 0.0

    if d == 0 or bf == 0 or t == 0:
        return _draw_fallback(section)

    prims: List[Primitive] = []

    y_top = d / 2
    y_bot = -d / 2
    x_left = -bf / 2
    x_right = bf / 2

    # Rectángulo exterior con esquinas redondeadas
    if R > 0:
        # Lados horizontales exteriores
        prims.append(Line(x1=x_left + R, y1=y_top, x2=x_right - R, y2=y_top))
        prims.append(Line(x1=x_left + R, y1=y_bot, x2=x_right - R, y2=y_bot))
        # Lados verticales exteriores
        prims.append(Line(x1=x_left, y1=y_top - R, x2=x_left, y2=y_bot + R))
        prims.append(Line(x1=x_right, y1=y_top - R, x2=x_right, y2=y_bot + R))
        # Esquinas exteriores
        prims.append(Arc(cx=x_left + R, cy=y_top - R, radius=R, start_angle_deg=90, span_angle_deg=90))
        prims.append(Arc(cx=x_right - R, cy=y_top - R, radius=R, start_angle_deg=0, span_angle_deg=90))
        prims.append(Arc(cx=x_left + R, cy=y_bot + R, radius=R, start_angle_deg=180, span_angle_deg=90))
        prims.append(Arc(cx=x_right - R, cy=y_bot + R, radius=R, start_angle_deg=270, span_angle_deg=90))
    else:
        # Rectángulo simple
        prims.append(Line(x1=x_left, y1=y_top, x2=x_right, y2=y_top))
        prims.append(Line(x1=x_right, y1=y_top, x2=x_right, y2=y_bot))
        prims.append(Line(x1=x_right, y1=y_bot, x2=x_left, y2=y_bot))
        prims.append(Line(x1=x_left, y1=y_bot, x2=x_left, y2=y_top))

    # Rectángulo interior
    y_top_in = y_top - t
    y_bot_in = y_bot + t
    x_left_in = x_left + t
    x_right_in = x_right - t
    R_in = max(0, R - t)

    if R_in > 0:
        prims.append(Line(x1=x_left_in + R_in, y1=y_top_in, x2=x_right_in - R_in, y2=y_top_in))
        prims.append(Line(x1=x_left_in + R_in, y1=y_bot_in, x2=x_right_in - R_in, y2=y_bot_in))
        prims.append(Line(x1=x_left_in, y1=y_top_in - R_in, x2=x_left_in, y2=y_bot_in + R_in))
        prims.append(Line(x1=x_right_in, y1=y_top_in - R_in, x2=x_right_in, y2=y_bot_in + R_in))
        prims.append(Arc(cx=x_left_in + R_in, cy=y_top_in - R_in, radius=R_in, start_angle_deg=90, span_angle_deg=90))
        prims.append(Arc(cx=x_right_in - R_in, cy=y_top_in - R_in, radius=R_in, start_angle_deg=0, span_angle_deg=90))
        prims.append(Arc(cx=x_left_in + R_in, cy=y_bot_in + R_in, radius=R_in, start_angle_deg=180, span_angle_deg=90))
        prims.append(Arc(cx=x_right_in - R_in, cy=y_bot_in + R_in, radius=R_in, start_angle_deg=270, span_angle_deg=90))
    else:
        prims.append(Line(x1=x_left_in, y1=y_top_in, x2=x_right_in, y2=y_top_in))
        prims.append(Line(x1=x_right_in, y1=y_top_in, x2=x_right_in, y2=y_bot_in))
        prims.append(Line(x1=x_right_in, y1=y_bot_in, x2=x_left_in, y2=y_bot_in))
        prims.append(Line(x1=x_left_in, y1=y_bot_in, x2=x_left_in, y2=y_top_in))

    # Cotas
    _add_vertical_dim(prims, y_bot, y_top, x_right + 20, f"d = {_fmt(d)}")
    _add_horizontal_dim(prims, x_left, x_right, y_top + 20, f"bf = {_fmt(bf)}")
    prims.append(Dimension(kind="horizontal", x1=x_right, y1=y_bot - 10, x2=x_right - t, y2=y_bot - 10,
                            text=f"t = {_fmt(t)}"))

    return prims


# ----------------------------------------------------------------------
# Plantilla: hss_circ (HSS circular, O tubos)
# ----------------------------------------------------------------------
def _draw_hss_circ(section: Section) -> List[Primitive]:
    """Dibuja un tubo circular (corona con diámetro exterior e interior)."""
    D = section.d or 0.0
    t = section.t_nom or section.tf or 0.0

    if D == 0:
        return _draw_fallback(section)

    prims: List[Primitive] = []

    r_ext = D / 2
    r_int = max(0, r_ext - t)

    # Círculo exterior
    prims.append(Circle(cx=0, cy=0, radius=r_ext))

    # Círculo interior
    if r_int > 0:
        prims.append(Circle(cx=0, cy=0, radius=r_int))

    # Cotas
    _add_vertical_dim(prims, -r_ext, r_ext, r_ext + 20, f"D = {_fmt(D)}")
    if t > 0:
        prims.append(Dimension(kind="horizontal", x1=r_int, y1=0, x2=r_ext, y2=0,
                                text=f"t = {_fmt(t)}"))

    # Centroide
    prims.append(Line(x1=-10, y1=0, x2=10, y2=0, layer="centerline"))
    prims.append(Line(x1=0, y1=-10, x2=0, y2=10, layer="centerline"))

    return prims


# ----------------------------------------------------------------------
# Plantilla: tee (WT, MT, ST, T)
# ----------------------------------------------------------------------
def _draw_tee(section: Section) -> List[Primitive]:
    """Dibuja un perfil tee (alma + 1 ala en la parte superior)."""
    d = section.d or 0.0
    bf = section.bf or 0.0
    tf = section.tf or 0.0
    tw = section.tw or 0.0
    r = section.r or 0.0

    if d == 0 or bf == 0:
        return _draw_fallback(section)

    prims: List[Primitive] = []

    # Tee: ala en la parte superior, alma hacia abajo
    # El origen está en la base del alma (no en el centroide)
    y_top = d  # cara superior del ala
    y_bot = 0  # base del alma
    y_ala_in = y_top - tf
    x_left = -bf / 2
    x_right = bf / 2
    x_left_web = -tw / 2
    x_right_web = tw / 2

    # Ala superior
    prims.append(Line(x1=x_left, y1=y_top, x2=x_right, y2=y_top))
    prims.append(Line(x1=x_right, y1=y_top, x2=x_right, y2=y_ala_in))
    prims.append(Line(x1=x_right, y1=y_ala_in, x2=x_right_web + r, y2=y_ala_in))
    if r > 0:
        prims.append(Arc(cx=x_right_web + r, cy=y_ala_in - r, radius=r,
                          start_angle_deg=90, span_angle_deg=-90))
        prims.append(Arc(cx=x_left_web - r, cy=y_ala_in - r, radius=r,
                          start_angle_deg=0, span_angle_deg=-90))
    prims.append(Line(x1=x_left_web - r, y1=y_ala_in, x2=x_left, y2=y_ala_in))
    prims.append(Line(x1=x_left, y1=y_ala_in, x2=x_left, y2=y_top))

    # Alma (hacia abajo desde el ala)
    prims.append(Line(x1=x_right_web, y1=y_ala_in - r, x2=x_right_web, y2=y_bot))
    prims.append(Line(x1=x_left_web, y1=y_ala_in - r, x2=x_left_web, y2=y_bot))
    prims.append(Line(x1=x_left_web, y1=y_bot, x2=x_right_web, y2=y_bot))

    # Cotas
    _add_vertical_dim(prims, y_bot, y_top, x_right + 20, f"d = {_fmt(d)}")
    _add_horizontal_dim(prims, x_left, x_right, y_top + 20, f"bf = {_fmt(bf)}")
    prims.append(Dimension(kind="horizontal", x1=-tw/2, y1=y_bot - 10, x2=tw/2, y2=y_bot - 10,
                            text=f"tw = {_fmt(tw)}"))

    return prims


# ----------------------------------------------------------------------
# Fallback: rectángulo simple
# ----------------------------------------------------------------------
def _draw_fallback(section: Section) -> List[Primitive]:
    """Dibuja un rectángulo simple cuando no hay suficiente información."""
    d = section.d or 100.0
    bf = section.bf or 50.0
    prims: List[Primitive] = []
    prims.append(Line(x1=-bf/2, y1=-d/2, x2=bf/2, y2=-d/2))
    prims.append(Line(x1=bf/2, y1=-d/2, x2=bf/2, y2=d/2))
    prims.append(Line(x1=bf/2, y1=d/2, x2=-bf/2, y2=d/2))
    prims.append(Line(x1=-bf/2, y1=d/2, x2=-bf/2, y2=-d/2))
    prims.append(Text(x=0, y=0, text=section.designation_modern, size=14))
    return prims


# ----------------------------------------------------------------------
# Bounding box
# ----------------------------------------------------------------------
def bounding_box(primitives: List[Primitive]) -> Tuple[float, float, float, float]:
    """Retorna (xmin, ymin, xmax, ymax) de todas las primitivas."""
    if not primitives:
        return (-100, -100, 100, 100)

    xmin = ymin = float('inf')
    xmax = ymax = float('-inf')

    for p in primitives:
        if isinstance(p, Line):
            xmin = min(xmin, p.x1, p.x2)
            xmax = max(xmax, p.x1, p.x2)
            ymin = min(ymin, p.y1, p.y2)
            ymax = max(ymax, p.y1, p.y2)
        elif isinstance(p, (Arc, Circle)):
            xmin = min(xmin, p.cx - p.radius)
            xmax = max(xmax, p.cx + p.radius)
            ymin = min(ymin, p.cy - p.radius)
            ymax = max(ymax, p.cy + p.radius)
        elif isinstance(p, Dimension):
            xmin = min(xmin, p.x1, p.x2)
            xmax = max(xmax, p.x1, p.x2)
            ymin = min(ymin, p.y1, p.y2)
            ymax = max(ymax, p.y1, p.y2)
        elif isinstance(p, Text):
            xmin = min(xmin, p.x)
            xmax = max(xmax, p.x)
            ymin = min(ymin, p.y)
            ymax = max(ymax, p.y)

    # Padding
    pad = 30.0
    return (xmin - pad, ymin - pad, xmax + pad, ymax + pad)
