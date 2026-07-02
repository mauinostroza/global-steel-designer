"""
Editor — cálculo de propiedades para perfiles custom.

Contiene `CustomSectionCalculator` con fórmulas clásicas de resistencia
de materiales para calcular propiedades geométricas, inerciales y de
torsión desde dimensiones básicas.

Tipos soportados:
    i_welded  — perfil I soldado (d, bf, tf, tw)
    hss_rect  — cajón rectangular soldado (B, D, t)
    hss_circ  — tubo circular (D, t)
    angle     — ángulo L (D, B, t)

También incluye validación de coherencia dimensional.

Uso:
    from suite_steel_catalog.editor import CustomSectionCalculator, ValidationResult
    props = CustomSectionCalculator.i_welded(d=200, bf=100, tf=8, tw=5)
    validation = CustomSectionCalculator.validate(d=200, bf=100, tf=8, tw=5)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ----------------------------------------------------------------------
# Resultado del cálculo
# ----------------------------------------------------------------------
@dataclass
class CalculatedProperties:
    """Propiedades calculadas desde dimensiones básicas.

    Todas en unidades SI (mm, mm², mm⁴, mm³, kg/m, MPa).
    """
    # Geometría
    area_mm2: float = 0.0
    weight_kg_m: float = 0.0
    perimeter_mm: float = 0.0
    is_hollow: bool = False
    is_sym_x: bool = True
    is_sym_y: bool = True

    # Eje fuerte X-X
    Ix_mm4: float = 0.0
    Sx_mm3: float = 0.0
    Zx_mm3: float = 0.0
    rx_mm: float = 0.0

    # Eje débil Y-Y
    Iy_mm4: float = 0.0
    Sy_mm3: float = 0.0
    Zy_mm3: float = 0.0
    ry_mm: float = 0.0

    # Torsión
    J_mm4: float = 0.0
    Cw_mm6: float = 0.0

    # Pandeo local
    bf_2tf: Optional[float] = None
    h_tw: Optional[float] = None
    b_t: Optional[float] = None
    D_t: Optional[float] = None


# ----------------------------------------------------------------------
# Resultado de validación
# ----------------------------------------------------------------------
@dataclass
class ValidationResult:
    """Resultado de validar dimensiones de un perfil custom."""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True si no hay errores (warnings son aceptables)."""
        return len(self.errors) == 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


# ----------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------
STEEL_DENSITY_KG_M3 = 7850.0


# ----------------------------------------------------------------------
# Calculador
# ----------------------------------------------------------------------
class CustomSectionCalculator:
    """Calcula propiedades de secciones custom desde dimensiones básicas.

    Todos los métodos son estáticos y no tienen estado.
    """

    # ------------------------------------------------------------------
    # Perfil I soldado
    # ------------------------------------------------------------------
    @staticmethod
    def i_welded(d: float, bf: float, tf: float, tw: float) -> CalculatedProperties:
        """Calcula propiedades de un perfil I soldado (simétrico).

        Args:
            d: peralto total (mm)
            bf: ancho de ala (mm)
            tf: espesor de ala (mm)
            tw: espesor de alma (mm)

        Returns:
            CalculatedProperties con todas las propiedades.
        """
        h = d - 2 * tf  # peralto claro del alma

        # Área
        A = 2 * bf * tf + h * tw
        peso = A * STEEL_DENSITY_KG_M3 * 1e-6  # kg/m
        perimeter = 2 * (bf + d) + 2 * (h - tw)  # aproximado

        # Inercia eje fuerte (paralelo al alma)
        # Ix = (bf * d³)/12 - ((bf - tw) * h³)/12
        Ix = (bf * d**3) / 12 - ((bf - tw) * h**3) / 12
        Sx = Ix / (d / 2)
        # Módulo plástico: Zx = bf*d²/2 - (bf-tw)*h²/4 (aproximado para simétrico)
        # Más preciso: Zx = (bf * d²)/4 - ((bf - tw) * h²) / 4
        # Pero la fórmula exacta considera el eje plástico en d/2:
        # Zx = 2 * [bf * tf * (d/2 - tf/2) + (h/2) * tw * (h/4)]
        #    = bf * tf * (d - tf) + tw * h² / 4
        Zx = bf * tf * (d - tf) + tw * h**2 / 4
        rx = math.sqrt(Ix / A) if A > 0 else 0

        # Inercia eje débil (paralelo al ala)
        # Iy = 2 * (tf * bf³)/12 + (h * tw³)/12
        Iy = 2 * (tf * bf**3) / 12 + (h * tw**3) / 12
        Sy = Iy / (bf / 2)
        # Zy = (tf * bf²)/2 + (h * tw²)/4 ... pero más preciso:
        # Zy = bf² * tf / 2 + ... (aproximado)
        Zy = tf * bf**2 / 2 + h * tw**2 / 4  # aproximado (no es exacto pero cercano)
        ry = math.sqrt(Iy / A) if A > 0 else 0

        # Torsión (sección abierta: J = (1/3) * Σ(b*t³))
        J = (1/3) * (2 * bf * tf**3 + h * tw**3)
        # Constante de alabeo (sección I bicisimétrica)
        # Cw = (tf * bf³ * h_int²) / 24  donde h_int = d - tf
        h_int = d - tf
        Cw = (tf * bf**3 * h_int**2) / 24

        # Esbeltez
        bf_2tf = bf / (2 * tf) if tf > 0 else None
        h_tw = h / tw if tw > 0 else None

        return CalculatedProperties(
            area_mm2=A,
            weight_kg_m=peso,
            perimeter_mm=perimeter,
            is_hollow=False,
            is_sym_x=True,
            is_sym_y=True,
            Ix_mm4=Ix,
            Sx_mm3=Sx,
            Zx_mm3=Zx,
            rx_mm=rx,
            Iy_mm4=Iy,
            Sy_mm3=Sy,
            Zy_mm3=Zy,
            ry_mm=ry,
            J_mm4=J,
            Cw_mm6=Cw,
            bf_2tf=bf_2tf,
            h_tw=h_tw,
        )

    # ------------------------------------------------------------------
    # Cajón rectangular hueco
    # ------------------------------------------------------------------
    @staticmethod
    def hss_rect(B: float, D: float, t: float) -> CalculatedProperties:
        """Calcula propiedades de un cajón rectangular soldado.

        Args:
            B: ancho total (mm)
            D: peralto total (mm)
            t: espesor (mm)
        """
        # Área = área exterior - área interior
        A_ext = B * D
        A_int = (B - 2*t) * (D - 2*t)
        A = A_ext - A_int
        peso = A * STEEL_DENSITY_KG_M3 * 1e-6
        perimeter = 2 * (B + D)

        # Inercias (rectángulo hueco)
        # Ix = (B * D³ - b_int * d_int³) / 12
        b_int = B - 2*t
        d_int = D - 2*t
        Ix = (B * D**3 - b_int * d_int**3) / 12
        Iy = (D * B**3 - d_int * b_int**3) / 12
        Sx = Ix / (D / 2)
        Sy = Iy / (B / 2)
        # Módulo plástico (rectángulo hueco)
        Zx = (B * D**2 / 4) - (b_int * d_int**2 / 4)
        Zy = (D * B**2 / 4) - (d_int * b_int**2 / 4)
        rx = math.sqrt(Ix / A) if A > 0 else 0
        ry = math.sqrt(Iy / A) if A > 0 else 0

        # Torsión (sección cerrada: fórmula de Bredt-Batho)
        # A_mid = área encerrada por la línea media
        B_mid = B - t
        D_mid = D - t
        A_mid = B_mid * D_mid
        s_mid = 2 * (B_mid + D_mid)  # perímetro de la línea media
        J = 4 * A_mid**2 * t / s_mid if s_mid > 0 else 0
        # Cw ≈ 0 para sección cerrada
        Cw = 0.0

        # Esbeltez
        b_t = (B - 2*t) / t if t > 0 else None  # ancho claro / espesor
        h_t = (D - 2*t) / t if t > 0 else None  # peralto claro / espesor

        return CalculatedProperties(
            area_mm2=A,
            weight_kg_m=peso,
            perimeter_mm=perimeter,
            is_hollow=True,
            is_sym_x=True,
            is_sym_y=True,
            Ix_mm4=Ix,
            Sx_mm3=Sx,
            Zx_mm3=Zx,
            rx_mm=rx,
            Iy_mm4=Iy,
            Sy_mm3=Sy,
            Zy_mm3=Zy,
            ry_mm=ry,
            J_mm4=J,
            Cw_mm6=Cw,
            b_t=b_t,
            h_tw=h_t,
        )

    # ------------------------------------------------------------------
    # Tubo circular
    # ------------------------------------------------------------------
    @staticmethod
    def hss_circ(D: float, t: float) -> CalculatedProperties:
        """Calcula propiedades de un tubo circular.

        Args:
            D: diámetro exterior (mm)
            t: espesor (mm)
        """
        r_ext = D / 2
        r_int = (D - 2*t) / 2
        # Área
        A = math.pi * (r_ext**2 - r_int**2)
        peso = A * STEEL_DENSITY_KG_M3 * 1e-6
        perimeter = math.pi * D

        # Inercia (corona circular)
        I = (math.pi / 4) * (r_ext**4 - r_int**4)
        S = I / r_ext
        # Módulo plástico (corona)
        Z = (4/3) * (r_ext**3 - r_int**3)
        r = math.sqrt(I / A) if A > 0 else 0

        # Torsión (sección circular cerrada): J = 2 * I
        J = 2 * I
        Cw = 0.0  # sección circular: Cw = 0

        # Esbeltez
        D_t = D / t if t > 0 else None

        return CalculatedProperties(
            area_mm2=A,
            weight_kg_m=peso,
            perimeter_mm=perimeter,
            is_hollow=True,
            is_sym_x=True,
            is_sym_y=True,
            Ix_mm4=I,
            Sx_mm3=S,
            Zx_mm3=Z,
            rx_mm=r,
            Iy_mm4=I,  # circular: Ix = Iy
            Sy_mm3=S,
            Zy_mm3=Z,
            ry_mm=r,
            J_mm4=J,
            Cw_mm6=Cw,
            D_t=D_t,
        )

    # ------------------------------------------------------------------
    # Ángulo L
    # ------------------------------------------------------------------
    @staticmethod
    def angle(D: float, B: float, t: float) -> CalculatedProperties:
        """Calcula propiedades de un ángulo L (simétrico o asimétrico).

        Args:
            D: lado mayor (mm) — altura
            B: lado menor (mm) — ancho (puede ser igual a D para L simétrica)
            t: espesor (mm)
        """
        # Área = D*t + B*t - t²
        A = D * t + B * t - t**2
        peso = A * STEEL_DENSITY_KG_M3 * 1e-6
        perimeter = 2 * (D + B) - 2 * t  # aproximado

        # Centroides (desde la esquina inferior-izquierda)
        # x_bar = (B*t*(B/2) + (D-t)*t*(t/2)) / A  ... no, es:
        # Para L: ala horizontal arriba (B x t), alma vertical izquierda (D x t)
        # Pero en nuestra convención, D es la altura, B es el ancho del ala
        # Centroides desde la esquina inferior-izquierda exterior:
        # x_bar = (B² + B*t - t²) / (2*(B + D - t))
        # y_bar = (D² + D*t - t²) / (2*(B + D - t))
        x_bar = (B**2 + B*t - t**2) / (2 * (B + D - t))
        y_bar = (D**2 + D*t - t**2) / (2 * (B + D - t))

        # Inercias respecto a ejes paralelos a los lados (pasando por la esquina)
        # Ix_corner = (t * D³)/3 + ((B-t) * t³)/3
        Ix_corner = (t * D**3) / 3 + ((B - t) * t**3) / 3
        Iy_corner = (t * B**3) / 3 + ((D - t) * t**3) / 3

        # Teorema de ejes paralelos: I_centroidal = I_corner - A * d²
        Ix = Ix_corner - A * y_bar**2
        Iy = Iy_corner - A * x_bar**2

        # Módulos elásticos (respecto a las fibras extremas)
        # Fibra superior: y_top = D - y_bar
        # Fibra inferior: y_bot = y_bar
        # Sx = Ix / max(y_top, y_bot)
        y_top = D - y_bar
        y_bot = y_bar
        Sx = Ix / max(y_top, y_bot) if max(y_top, y_bot) > 0 else 0
        # Fibra derecha: x_right = B - x_bar
        # Fibra izquierda: x_left = x_bar
        x_right = B - x_bar
        x_left = x_bar
        Sy = Iy / max(x_right, x_left) if max(x_right, x_left) > 0 else 0

        # Radios de giro
        rx = math.sqrt(Ix / A) if A > 0 else 0
        ry = math.sqrt(Iy / A) if A > 0 else 0

        # Torsión (sección abierta: J = (1/3) * Σ(b*t³))
        # Para L: dos ramas de espesor t
        J = (1/3) * (D * t**3 + (B - t) * t**3)
        Cw = 0.0  # L tiene Cw pequeño, lo dejamos en 0

        # Esbeltez
        b_t = B / t if t > 0 else None
        D_t = D / t if t > 0 else None

        # Centroides en mm desde el centro geométrico (para que la UI lo use)
        # El centro geométrico es (B/2, D/2)
        # centroid_x_mm = x_bar - B/2
        # centroid_y_mm = y_bar - D/2

        return CalculatedProperties(
            area_mm2=A,
            weight_kg_m=peso,
            perimeter_mm=perimeter,
            is_hollow=False,
            is_sym_x=False,  # L no es simétrica
            is_sym_y=False,
            Ix_mm4=Ix,
            Sx_mm3=Sx,
            Zx_mm3=Sx * 1.15,  # aproximación: Zx ≈ 1.15 * Sx para L
            rx_mm=rx,
            Iy_mm4=Iy,
            Sy_mm3=Sy,
            Zy_mm3=Sy * 1.15,
            ry_mm=ry,
            J_mm4=J,
            Cw_mm6=Cw,
            b_t=b_t,
            D_t=D_t,
        )

    # ------------------------------------------------------------------
    # Validación
    # ------------------------------------------------------------------
    @staticmethod
    def validate(d: float, bf: float, tf: float, tw: float,
                 profile_type: str = "i_welded") -> ValidationResult:
        """Valida dimensiones de un perfil custom.

        Args:
            d: peralto (mm)
            bf: ancho (mm)
            tf: espesor ala o general (mm)
            tw: espesor alma (mm, solo para i_welded)
            profile_type: 'i_welded' | 'hss_rect' | 'hss_circ' | 'angle'

        Returns:
            ValidationResult con errors (bloqueantes) y warnings (no bloqueantes).
        """
        result = ValidationResult()

        # Validaciones universales
        if d is None or d <= 0:
            result.add_error("El peralto d debe ser positivo")
        if bf is None or bf <= 0:
            result.add_error("El ancho bf debe ser positivo")
        if tf is None or tf <= 0:
            result.add_error("El espesor tf debe ser positivo")

        if result.errors:
            return result  # No continuar si hay errores básicos

        # Validaciones por tipo
        if profile_type == "i_welded":
            if tw is None or tw <= 0:
                result.add_error("El espesor del alma tw debe ser positivo")
            else:
                if d <= 2 * tf:
                    result.add_error(f"d ({d}) debe ser > 2*tf ({2*tf})")
                if bf <= tw:
                    result.add_warning(f"bf ({bf}) debería ser > tw ({tw})")
                # Esbeltez razonable
                if tf > 0:
                    bf_2tf = bf / (2 * tf)
                    if bf_2tf > 60:
                        result.add_warning(f"bf/2tf={bf_2tf:.1f} > 60 (ala muy esbelta)")
                    h_tw = (d - 2*tf) / tw
                    if h_tw > 200:
                        result.add_warning(f"h/tw={h_tw:.1f} > 200 (alma muy esbelta)")

        elif profile_type == "hss_rect":
            if d <= 2 * tf:
                result.add_error(f"D ({d}) debe ser > 2*t ({2*tf})")
            if bf <= 2 * tf:
                result.add_error(f"B ({bf}) debe ser > 2*t ({2*tf})")
            # Esbeltez
            b_t = (bf - 2*tf) / tf
            if b_t > 50:
                result.add_warning(f"b/t={b_t:.1f} > 50 (pared muy esbelta)")

        elif profile_type == "hss_circ":
            if d <= 2 * tf:
                result.add_error(f"D ({d}) debe ser > 2*t ({2*tf})")
            D_t = d / tf
            if D_t > 100:
                result.add_warning(f"D/t={D_t:.1f} > 100 (pared muy esbelta)")

        elif profile_type == "angle":
            if tf > d:
                result.add_error(f"t ({tf}) no puede ser > D ({d})")
            if tf > bf:
                result.add_error(f"t ({tf}) no puede ser > B ({bf})")
            b_t = bf / tf
            if b_t > 30:
                result.add_warning(f"b/t={b_t:.1f} > 30 (ala muy esbelta)")

        return result

    # ------------------------------------------------------------------
    # Cálculo de peso teórico
    # ------------------------------------------------------------------
    @staticmethod
    def area_to_weight_kg_m(area_mm2: float) -> float:
        """Convierte área (mm²) a peso (kg/m) usando densidad del acero."""
        return area_mm2 * STEEL_DENSITY_KG_M3 * 1e-6
