"""
Dataclasses del catálogo: Family, Material, Section.

Estas son las entidades principales que la librería expone a las apps
consumidoras. Son inmutables excepto para los campos de materiales
(asociación perezosa).

Todas las longitudes en mm, áreas en mm², inercias en mm⁴, módulos en mm³,
peso en kg/m, tensiones en MPa.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .catalog import Catalog


@dataclass(frozen=True, slots=True)
class Family:
    """Familia de perfiles (IN, HN, W, IPE, HEA, ...)."""
    family_code: str
    name_es: str
    name_en: Optional[str]
    manufacturing_process: str   # 'hot_rolled' | 'cold_formed' | 'welded' | 'back_to_back' | 'cut_from_parent'
    source_standard: str         # 'ICHA' | 'AISC 360' | 'EN 10365' | 'CINTAC' | ...
    source_edition: Optional[str]
    is_chilean: bool
    has_subclasses: bool
    drawing_template: str        # 'I_welded' | 'I_rolled' | 'channel_rolled' | ...

    @classmethod
    def from_row(cls, row: sqlite3.Row | tuple) -> "Family":
        if isinstance(row, sqlite3.Row):
            row = tuple(row)
        return cls(
            family_code=row[0],
            name_es=row[1],
            name_en=row[2],
            manufacturing_process=row[3],
            source_standard=row[4],
            source_edition=row[5],
            is_chilean=bool(row[6]),
            has_subclasses=bool(row[7]),
            drawing_template=row[8],
        )

    def to_dict(self) -> dict:
        return {
            "family_code": self.family_code,
            "name_es": self.name_es,
            "name_en": self.name_en,
            "manufacturing_process": self.manufacturing_process,
            "source_standard": self.source_standard,
            "source_edition": self.source_edition,
            "is_chilean": self.is_chilean,
            "has_subclasses": self.has_subclasses,
            "drawing_template": self.drawing_template,
        }


@dataclass(frozen=True, slots=True)
class Material:
    """Grado de acero."""
    material_code: str
    standard: str
    grade: Optional[str]
    Fy_MPa: float
    Fu_MPa: Optional[float] = None
    E_MPa: float = 200_000.0
    nu: float = 0.30
    applicable_families: Optional[str] = None  # CSV: 'IN,HN,IP,IE,PH'

    @classmethod
    def from_row(cls, row: sqlite3.Row | tuple) -> "Material":
        if isinstance(row, sqlite3.Row):
            row = tuple(row)
        return cls(
            material_code=row[0],
            standard=row[1],
            grade=row[2],
            Fy_MPa=float(row[3]),
            Fu_MPa=float(row[4]) if row[4] is not None else None,
            E_MPa=float(row[5]) if row[5] is not None else 200_000.0,
            nu=float(row[6]) if row[6] is not None else 0.30,
            applicable_families=row[7],
        )

    def to_dict(self) -> dict:
        return {
            "material_code": self.material_code,
            "standard": self.standard,
            "grade": self.grade,
            "Fy_MPa": self.Fy_MPa,
            "Fu_MPa": self.Fu_MPa,
            "E_MPa": self.E_MPa,
            "nu": self.nu,
            "applicable_families": self.applicable_families,
        }


@dataclass(slots=True)
class Section:
    """Perfil estructural de acero.

    `frozen=False` porque los materiales se cargan perezosamente.
    Los campos numéricos son Optional porque no todas las familias
    tienen todas las propiedades (p.ej. J y Cw solo existen en L y Cajón).
    """
    # Identidad
    section_id: int
    family: Family
    subclass: Optional[str]
    designation_modern: str
    designation_legacy: Optional[str]
    designation_aisc: Optional[str]
    designation_en: Optional[str]
    source_catalog: str
    source_edition: Optional[str]
    available_sack: bool
    available_cintac: bool
    is_standard: bool
    is_custom: bool
    notes: Optional[str]

    # Geometría (mm)
    d: Optional[float]
    bf: Optional[float]
    tf: Optional[float]
    tw: Optional[float]
    h: Optional[float]
    T_clear: Optional[float]
    k: Optional[float]
    k_design: Optional[float]
    k_detail: Optional[float]
    r: Optional[float]
    R_ext: Optional[float]
    R_int: Optional[float]
    B: Optional[float]
    C_dim: Optional[float]
    t_nom: Optional[float]
    t_des: Optional[float]
    area_mm2: float
    perimeter_mm: Optional[float]
    weight_kg_m: float
    is_hollow: bool
    is_sym_x: bool
    is_sym_y: bool
    centroid_x_mm: float
    centroid_y_mm: float

    # Eje fuerte X-X
    Ix_mm4: Optional[float]
    Sx_mm3: Optional[float]
    Zx_mm3: Optional[float]
    rx_mm: Optional[float]
    xp_mm: Optional[float]

    # Eje débil Y-Y
    Iy_mm4: Optional[float]
    Sy_mm3: Optional[float]
    Zy_mm3: Optional[float]
    ry_mm: Optional[float]
    yp_mm: Optional[float]

    # Torsión
    J_mm4: Optional[float]
    Cw_mm6: Optional[float]
    Wno: Optional[float]
    Sw: Optional[float]
    Qf: Optional[float]
    Qw: Optional[float]
    H_const: Optional[float]
    ro_mm: Optional[float]
    xo_mm: Optional[float]
    io_mm: Optional[float]
    beta: Optional[float]
    j: Optional[float]

    # Pandeo local
    bf_2tf: Optional[float]
    h_tw: Optional[float]
    b_t: Optional[float]
    D_t: Optional[float]
    Qs: Optional[float]
    Qa: Optional[float]
    ia: Optional[float]
    it: Optional[float]
    X1: Optional[float]
    Fy_default_MPa: Optional[float]

    # Materiales (cargados perezosamente)
    materials: list[Material] = field(default_factory=list)
    _default_material: Optional[Material] = None
    _materials_loaded: bool = False

    # ------------------------------------------------------------------
    # Constructores
    # ------------------------------------------------------------------
    @classmethod
    def from_row(cls, row: sqlite3.Row | tuple, family: Family) -> "Section":
        """Construye una Section a partir de una fila SQL con todas las
        columnas relevantes (producto de un JOIN multi-tabla).

        Orden esperado de columnas (ver Repository._section_select_sql).
        """
        if isinstance(row, sqlite3.Row):
            row = tuple(row)
        # mapeo por índice. La estructura del SELECT se controla en
        # Repository, así que es seguro.
        return cls(
            family=family,
            section_id=row[0],
            subclass=row[2],
            designation_modern=row[3],
            designation_legacy=row[4],
            designation_aisc=row[5],
            designation_en=row[6],
            source_catalog=row[7],
            source_edition=row[8],
            available_sack=bool(row[9]),
            available_cintac=bool(row[10]),
            is_standard=bool(row[11]),
            is_custom=bool(row[12]),
            notes=row[13],
            d=row[14],
            bf=row[15],
            tf=row[16],
            tw=row[17],
            h=row[18],
            T_clear=row[19],
            k=row[20],
            k_design=row[21],
            k_detail=row[22],
            r=row[23],
            R_ext=row[24],
            R_int=row[25],
            B=row[26],
            C_dim=row[27],
            t_nom=row[28],
            t_des=row[29],
            area_mm2=float(row[30]) if row[30] is not None else 0.0,
            perimeter_mm=row[31],
            weight_kg_m=float(row[32]) if row[32] is not None else 0.0,
            is_hollow=bool(row[33]),
            is_sym_x=bool(row[34]),
            is_sym_y=bool(row[35]),
            centroid_x_mm=float(row[36]) if row[36] is not None else 0.0,
            centroid_y_mm=float(row[37]) if row[37] is not None else 0.0,
            Ix_mm4=row[38],
            Sx_mm3=row[39],
            Zx_mm3=row[40],
            rx_mm=row[41],
            xp_mm=row[42],
            Iy_mm4=row[43],
            Sy_mm3=row[44],
            Zy_mm3=row[45],
            ry_mm=row[46],
            yp_mm=row[47],
            J_mm4=row[48],
            Cw_mm6=row[49],
            Wno=row[50],
            Sw=row[51],
            Qf=row[52],
            Qw=row[53],
            H_const=row[54],
            ro_mm=row[55],
            xo_mm=row[56],
            io_mm=row[57],
            beta=row[58],
            j=row[59],
            bf_2tf=row[60],
            h_tw=row[61],
            b_t=row[62],
            D_t=row[63],
            Qs=row[64],
            Qa=row[65],
            ia=row[66],
            it=row[67],
            X1=row[68],
            Fy_default_MPa=row[69],
        )

    # ------------------------------------------------------------------
    # Propiedades derivadas
    # ------------------------------------------------------------------
    @property
    def material(self) -> Optional[Material]:
        """Material por defecto del perfil."""
        return self._default_material

    @property
    def designation(self) -> str:
        """Designación preferida para mostrar al usuario."""
        return self.designation_modern

    @property
    def is_hollow_section(self) -> bool:
        """True si es una sección cerrada (HSS, Cajón, Tubo)."""
        return self.is_hollow

    # ------------------------------------------------------------------
    # Conversión a/desde dict (para serialización JSON)
    # ------------------------------------------------------------------
    def to_dict(self, include_materials: bool = True) -> dict:
        """Serializa a diccionario (JSON-ready).

        Args:
            include_materials: si True, incluye la lista de materiales.
                La carga de materiales es perezosa; si aún no se han
                cargado, se incluye una lista vacía.
        """
        d = {
            "section_id": self.section_id,
            "family_code": self.family.family_code,
            "subclass": self.subclass,
            "designation_modern": self.designation_modern,
            "designation_legacy": self.designation_legacy,
            "designation_aisc": self.designation_aisc,
            "designation_en": self.designation_en,
            "source_catalog": self.source_catalog,
            "source_edition": self.source_edition,
            "available_sack": self.available_sack,
            "available_cintac": self.available_cintac,
            "is_standard": self.is_standard,
            "is_custom": self.is_custom,
            "notes": self.notes,
            "d_mm": self.d,
            "bf_mm": self.bf,
            "tf_mm": self.tf,
            "tw_mm": self.tw,
            "h_mm": self.h,
            "T_clear_mm": self.T_clear,
            "k_mm": self.k,
            "k_design_mm": self.k_design,
            "k_detail_mm": self.k_detail,
            "r_mm": self.r,
            "R_ext_mm": self.R_ext,
            "R_int_mm": self.R_int,
            "B_mm": self.B,
            "C_dim_mm": self.C_dim,
            "t_nom_mm": self.t_nom,
            "t_des_mm": self.t_des,
            "area_mm2": self.area_mm2,
            "perimeter_mm": self.perimeter_mm,
            "weight_kg_m": self.weight_kg_m,
            "is_hollow": self.is_hollow,
            "is_sym_x": self.is_sym_x,
            "is_sym_y": self.is_sym_y,
            "centroid_x_mm": self.centroid_x_mm,
            "centroid_y_mm": self.centroid_y_mm,
            "Ix_mm4": self.Ix_mm4,
            "Sx_mm3": self.Sx_mm3,
            "Zx_mm3": self.Zx_mm3,
            "rx_mm": self.rx_mm,
            "xp_mm": self.xp_mm,
            "Iy_mm4": self.Iy_mm4,
            "Sy_mm3": self.Sy_mm3,
            "Zy_mm3": self.Zy_mm3,
            "ry_mm": self.ry_mm,
            "yp_mm": self.yp_mm,
            "J_mm4": self.J_mm4,
            "Cw_mm6": self.Cw_mm6,
            "Wno": self.Wno,
            "Sw": self.Sw,
            "Qf": self.Qf,
            "Qw": self.Qw,
            "H_const": self.H_const,
            "ro_mm": self.ro_mm,
            "xo_mm": self.xo_mm,
            "io_mm": self.io_mm,
            "beta": self.beta,
            "j": self.j,
            "bf_2tf": self.bf_2tf,
            "h_tw": self.h_tw,
            "b_t": self.b_t,
            "D_t": self.D_t,
            "Qs": self.Qs,
            "Qa": self.Qa,
            "ia": self.ia,
            "it": self.it,
            "X1": self.X1,
            "Fy_default_MPa": self.Fy_default_MPa,
        }
        if include_materials:
            d["materials"] = [m.to_dict() for m in self.materials]
            d["default_material"] = self._default_material.to_dict() if self._default_material else None
        return d

    def __str__(self) -> str:
        return f"{self.family.family_code} {self.designation_modern}"

    def __repr__(self) -> str:
        return (f"Section(id={self.section_id}, family='{self.family.family_code}', "
                f"designation='{self.designation_modern}', d={self.d}, "
                f"weight={self.weight_kg_m} kg/m)")
