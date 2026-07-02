"""
Conversiones de unidades para el catálogo de secciones.

La BD almacena todo en SI (mm, mm², mm⁴, mm³, mm⁶, kg/m, MPa).
Los parsers usan estos métodos al ingestar datos desde catálogos
que vienen en cm, pulg, kgf/m, ksi, etc.

Todos los métodos son estáticos y no tienen estado, lo que facilita
el testing unitario.
"""
from __future__ import annotations

import math


class Normalizer:
    """Conversores de unidades a SI."""

    # ------------------------------------------------------------------
    # Longitud → mm
    # ------------------------------------------------------------------
    @staticmethod
    def length_cm_to_mm(value_cm: float) -> float:
        """Centímetros → milímetros."""
        return value_cm * 10.0

    @staticmethod
    def length_m_to_mm(value_m: float) -> float:
        """Metros → milímetros."""
        return value_m * 1000.0

    @staticmethod
    def length_in_to_mm(value_in: float) -> float:
        """Pulgadas → milímetros (1 in = 25.4 mm exacto)."""
        return value_in * 25.4

    @staticmethod
    def length_ft_to_mm(value_ft: float) -> float:
        """Pies → milímetros (1 ft = 304.8 mm)."""
        return value_ft * 304.8

    # ------------------------------------------------------------------
    # Área → mm²
    # ------------------------------------------------------------------
    @staticmethod
    def area_cm2_to_mm2(value: float) -> float:
        """cm² → mm²."""
        return value * 100.0

    @staticmethod
    def area_m2_to_mm2(value: float) -> float:
        """m² → mm²."""
        return value * 1_000_000.0

    @staticmethod
    def area_in2_to_mm2(value: float) -> float:
        """pulg² → mm² (1 in² = 645.16 mm²)."""
        return value * 645.16

    # ------------------------------------------------------------------
    # Inercia → mm⁴
    # ------------------------------------------------------------------
    @staticmethod
    def inertia_cm4_to_mm4(value: float) -> float:
        """cm⁴ → mm⁴."""
        return value * 10_000.0

    @staticmethod
    def inertia_m4_to_mm4(value: float) -> float:
        """m⁴ → mm⁴."""
        return value * 1e12

    @staticmethod
    def inertia_in4_to_mm4(value: float) -> float:
        """pulg⁴ → mm⁴ (1 in⁴ = 416231.426 mm⁴)."""
        return value * 416_231.426

    # ------------------------------------------------------------------
    # Módulo de sección → mm³
    # ------------------------------------------------------------------
    @staticmethod
    def section_modulus_cm3_to_mm3(value: float) -> float:
        """cm³ → mm³."""
        return value * 1_000.0

    @staticmethod
    def section_modulus_in3_to_mm3(value: float) -> float:
        """pulg³ → mm³ (1 in³ = 16387.064 mm³)."""
        return value * 16_387.064

    # ------------------------------------------------------------------
    # Constante de alabeo → mm⁶
    # ------------------------------------------------------------------
    @staticmethod
    def warping_cm6_to_mm6(value: float) -> float:
        """cm⁶ → mm⁶."""
        return value * 1_000_000.0

    @staticmethod
    def warping_in6_to_mm6(value: float) -> float:
        """pulg⁶ → mm⁶ (1 in⁶ = 17720.91 mm⁶)."""
        return value * 17_720.91

    # ------------------------------------------------------------------
    # Peso → kg/m
    # ------------------------------------------------------------------
    @staticmethod
    def weight_lbf_ft_to_kg_m(value: float) -> float:
        """lbf/pie → kg/m (1 lbf/ft = 1.48816 kg/m)."""
        return value * 1.48816

    @staticmethod
    def weight_kgf_m_to_kg_m(value: float) -> float:
        """kgf/m → kg/m.

        En catálogo se asume equivalencia 1:1 (la diferencia entre
        kgf y kg en condiciones estándar de gravedad es < 0.5%).
        Los programas que necesitan kgf/m estricto pueden convertir
        en su propia capa.
        """
        return value

    @staticmethod
    def weight_kg_m_to_lbf_ft(value: float) -> float:
        """kg/m → lbf/pie (inverso, para exportación AISC)."""
        return value / 1.48816

    # ------------------------------------------------------------------
    # Tensión → MPa
    # ------------------------------------------------------------------
    @staticmethod
    def stress_ksi_to_MPa(value: float) -> float:
        """ksi → MPa (1 ksi = 6.89476 MPa)."""
        return value * 6.89476

    @staticmethod
    def stress_psi_to_MPa(value: float) -> float:
        """psi → MPa (1 psi = 0.00689476 MPa)."""
        return value * 0.00689476

    @staticmethod
    def stress_kgf_cm2_to_MPa(value: float) -> float:
        """kgf/cm² → MPa (1 kgf/cm² = 0.0980665 MPa)."""
        return value * 0.0980665

    @staticmethod
    def stress_N_mm2_to_MPa(value: float) -> float:
        """N/mm² → MPa (equivalencia 1:1)."""
        return value

    @staticmethod
    def stress_MPa_to_ksi(value: float) -> float:
        """MPa → ksi (inverso)."""
        return value / 6.89476

    # ------------------------------------------------------------------
    # Fuerza → kN
    # ------------------------------------------------------------------
    @staticmethod
    def force_kgf_to_kN(value: float) -> float:
        """kgf → kN (1 kgf = 0.00980665 kN)."""
        return value * 0.00980665

    @staticmethod
    def force_tonf_to_kN(value: float) -> float:
        """tonelada-fuerza (tf) → kN (1 tf = 9.80665 kN)."""
        return value * 9.80665

    @staticmethod
    def force_lbf_to_kN(value: float) -> float:
        """lbf → kN (1 lbf = 0.00444822 kN)."""
        return value * 0.00444822

    @staticmethod
    def force_kip_to_kN(value: float) -> float:
        """kip → kN (1 kip = 4.44822 kN)."""
        return value * 4.44822

    # ------------------------------------------------------------------
    # Momento → kN·m
    # ------------------------------------------------------------------
    @staticmethod
    def moment_kgf_m_to_kN_m(value: float) -> float:
        """kgf·m → kN·m."""
        return value * 0.00980665

    @staticmethod
    def moment_tonf_m_to_kN_m(value: float) -> float:
        """tf·m → kN·m (1 tf·m = 9.80665 kN·m)."""
        return value * 9.80665

    @staticmethod
    def moment_kip_ft_to_kN_m(value: float) -> float:
        """kip·ft → kN·m (1 kip·ft = 1.35582 kN·m)."""
        return value * 1.35582

    @staticmethod
    def moment_kip_in_to_kN_m(value: float) -> float:
        """kip·in → kN·m (1 kip·in = 0.112985 kN·m)."""
        return value * 0.112985

    # ------------------------------------------------------------------
    # Densidad de acero (constante)
    # ------------------------------------------------------------------
    STEEL_DENSITY_KG_M3: float = 7850.0
    GRAVITY_M_S2: float = 9.80665

    @classmethod
    def area_to_weight_kg_m(cls, area_mm2: float) -> float:
        """Calcula peso teórico (kg/m) a partir del área (mm²).

        Útil para validar consistencia: peso calculado ≈ peso declarado.
        """
        # area_mm2 * 1e-6 m²/mm² * 7850 kg/m³ = area_mm2 * 7850e-6 kg/m
        return area_mm2 * cls.STEEL_DENSITY_KG_M3 * 1e-6

    @classmethod
    def weight_to_area_mm2(cls, weight_kg_m: float) -> float:
        """Inverso de `area_to_weight_kg_m`."""
        return weight_kg_m / (cls.STEEL_DENSITY_KG_M3 * 1e-6)

    # ------------------------------------------------------------------
    # Utilidades de redondeo (los catálogos vienen con precisión variable)
    # ------------------------------------------------------------------
    @staticmethod
    def round_mm(value: float, decimals: int = 1) -> float:
        """Redondea una longitud en mm al número de decimales especificado."""
        if value is None or math.isnan(value):
            return value
        return round(value, decimals)

    @staticmethod
    def round_mm4(value: float, sig: int = 6) -> float:
        """Redondea una inercia en mm⁴ a `sig` cifras significativas."""
        if value is None or value == 0 or math.isnan(value):
            return value
        return round(value, sig - int(math.floor(math.log10(abs(value)))) - 1)
