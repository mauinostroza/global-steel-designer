"""Conexión SAP2000 OAPI — leer y modificar factores K en ángulos L.

Arquitectura:
  Sap2000Connector  →  maneja COM/OAPI (real o fake), unidades y metadatos
  AngleReader       →  lee selección, fuerzas (por combo/caso), largo, K
  AngleWriter       →  modifica K en SAP2000
  AngleService      →  orquesta: leer → calcular → escribir

Reglas seguidas (ver docs de integración SAP2000 del proyecto):
  - Nunca se lee un resultado sin fijar unidades explícitamente primero.
  - Nunca se leen fuerzas sin seleccionar explícitamente combo/caso de salida
    (si no se selecciona, se advierte en vez de asumir silenciosamente).
  - Los códigos de retorno != 0 se registran como advertencias, no se ocultan.
"""

from __future__ import annotations

import datetime
import math
import sys
from dataclasses import dataclass, field
from typing import Any, Optional


# ═══════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════

# Código de unidades de trabajo fijado por el programa: kN, mm, °C.
# (Convención OAPI SAP2000: 5 = kN_mm_C). Todo el resto del programa
# asume que las fuerzas llegan en kN y las longitudes en mm porque se
# fija este código explícitamente en cada conexión.
WORK_UNITS_CODE = 5
WORK_UNITS_LABEL = "kN, mm, °C"


@dataclass(slots=True)
class Sap2000ImportMeta:
    """Metadatos de la conexión/importación desde SAP2000."""
    version: str = ""
    model_filename: str = ""
    units_code: int = 0
    units_label: str = ""
    original_units_code: int = 0
    connected_at: str = ""
    warnings: list[str] = field(default_factory=list)

    def add_warning(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)


@dataclass(slots=True)
class AngleElementData:
    """Datos leídos desde SAP2000 para un elemento angular."""
    name: str               # nombre del frame (ej: "F1")
    section_name: str       # nombre de la sección (ej: "L100x100x10")
    b1: float               # pierna larga (mm)
    b2: float               # pierna corta (mm)
    t: float                # espesor (mm)
    length_mm: float        # largo del elemento (mm)
    conn_leg: str            # "long" o "short" (pierna conectada)
    conn_leg_is_default: bool  # True si no se pudo determinar y se usó el default
    Pu_comp_max: float      # máx compresión de todos los casos (kN, positivo)
    Pu_tens_max: float      # máx tracción de todos los casos (kN, positivo)
    critical_combo: str     # nombre del combo/caso que produjo Pu_comp_max
    K_strong_current: float # factor K actual (eje fuerte)
    K_weak_current: float   # factor K actual (eje débil)
    Fy_MPa: float           # tensión de fluencia
    E_MPa: float            # módulo de elasticidad


@dataclass(slots=True)
class KFactorResult:
    """Resultado del cálculo del nuevo factor K."""
    element: str
    section: str
    L_mm: float
    conn_leg: str
    KLr_eff: float          # (KL/r) efectivo calculado
    K_calculated: float     # nuevo K = (KL/r)ef / (L/r_conn)
    K_strong_old: float
    K_weak_old: float
    K_strong_new: float
    K_weak_new: float
    phiPn_kN: float
    Pu_comp_max: float
    Pu_tens_max: float
    ratio_comp: float | None
    ratio_tens: float | None
    ok_comp: bool | None
    ok_tens: bool | None
    critical_combo: str = ""
    calc_steps: list = field(default_factory=list)
    b1: float = 0.0
    b2: float = 0.0
    t: float = 0.0
    conn_leg_is_default: bool = False
    method: str = "LRFD"


# ═══════════════════════════════════════════════
#  SAP2000 CONNECTOR (COM/OAPI)
# ═══════════════════════════════════════════════


def _unwrap_ret(value: Any) -> tuple[Any, int]:
    """Normaliza el retorno de una función OAPI tipo "Get" que expone un
    valor de salida además del código de error: `valor` o `(valor, ret)`.

    Si el retorno es un escalar, se asume que ES el valor de salida (no el
    código de error) y que la llamada fue exitosa — válido solo para
    funciones "Get" cuyo único modo de fallo real se refleja en el valor
    (None/vacío) o en una excepción COM. NO usar con funciones "Set" que no
    tienen valor de salida propio: en esas, un escalar suelto ES el código
    de error, no un valor exitoso por defecto (ver `_setter_ret`).
    """
    if isinstance(value, (tuple, list)):
        if len(value) == 0:
            return None, -1
        return value[0], value[-1]
    return value, 0


def _setter_ret(value: Any) -> int:
    """Normaliza el retorno de una función OAPI tipo "Set" sin valor de
    salida propio: `ret` (int suelto) o `(..., ret)`.

    A diferencia de `_unwrap_ret`, aquí un escalar suelto ES el código de
    retorno real (no se asume éxito), porque estas funciones no tienen un
    "valor" adicional que exponer aparte del código de error.
    """
    if isinstance(value, (tuple, list)):
        return value[-1] if value else -1
    return value


class Sap2000Connector:
    """Maneja la conexión COM/OAPI con SAP2000.

    En Windows usa comtypes. Sin SAP2000 usa FakeSapObject para testing.
    """

    PROGIDS_V23 = [
        "CSI.SAP2000.API.SapObject.23",
        "CSI.SAP2000.API.SapObject",
        "CSI.SAP2000.API.SapObject.22",
        "CSI.SAP2000.API.SapObject.21",
    ]

    def __init__(self, use_fake: bool = False, progid: str | None = None):
        self._sap = None
        self._model = None
        self._use_fake = use_fake
        self._progid_override = progid
        self.meta = Sap2000ImportMeta()

    def connect(self) -> bool:
        """Conecta a una instancia ABIERTA de SAP2000 (nunca lanza una nueva).

        Estrategia:
          1. Si use_fake=True → FakeSapModel
          2. Probar TODOS los ProgIDs (específico primero, luego v23→genérico→v22→v21)
          3. Si ningún ProgID funciona → error, el usuario debe abrir SAP2000 manualmente

        Al conectar se fijan las unidades de trabajo (WORK_UNITS_CODE) y se
        registran metadatos (versión, archivo, unidades originales).
        """
        self.meta = Sap2000ImportMeta()

        if self._use_fake:
            self._sap = FakeSapObject()
            self._model = self._sap.SapModel
            self._set_units_and_metadata()
            return True

        try:
            import comtypes.client

            # Siempre probar TODOS los ProgIDs
            all_progids = list(self.PROGIDS_V23)

            # Si hay un progid específico, ponerlo primero
            if self._progid_override and self._progid_override in all_progids:
                all_progids.remove(self._progid_override)
                all_progids.insert(0, self._progid_override)
            elif self._progid_override and self._progid_override not in all_progids:
                all_progids.insert(0, self._progid_override)

            for progid in all_progids:
                try:
                    self._sap = comtypes.client.GetActiveObject(progid)
                    self._model = self._sap.SapModel
                    self._set_units_and_metadata()
                    return True
                except Exception:
                    continue

            print("Error: No hay instancia de SAP2000 abierta. Abre SAP2000 manualmente e intenta de nuevo.", file=sys.stderr)
            return False
        except Exception as e:
            print(f"Error conectando a SAP2000: {e}", file=sys.stderr)
            return False

    def _set_units_and_metadata(self) -> None:
        """Fija unidades de trabajo y registra metadatos del modelo.

        Regla obligatoria: nunca se leen resultados sin fijar unidades
        explícitamente primero. Cualquier fallo aquí queda registrado como
        advertencia visible, no se silencia.
        """
        self.meta.connected_at = datetime.datetime.now().isoformat(timespec="seconds")

        try:
            original, ret = _unwrap_ret(self._model.GetPresentUnits())
            if ret == 0 and original is not None:
                self.meta.original_units_code = int(original)
        except Exception as e:
            self.meta.add_warning(f"No se pudo leer unidades actuales: {e}")

        try:
            ret = _setter_ret(self._model.SetPresentUnits(WORK_UNITS_CODE))
            if ret != 0:
                self.meta.add_warning(
                    f"SetPresentUnits({WORK_UNITS_CODE}) retornó código {ret}; "
                    "las unidades pueden no ser las esperadas (kN, mm)."
                )
            self.meta.units_code = WORK_UNITS_CODE
            self.meta.units_label = WORK_UNITS_LABEL
        except Exception as e:
            self.meta.add_warning(f"No se pudo fijar unidades de trabajo: {e}")

        try:
            version, ret = _unwrap_ret(self._sap.GetVersion())
            if ret == 0 and version:
                self.meta.version = str(version)
        except Exception as e:
            self.meta.add_warning(f"No se pudo leer versión de SAP2000: {e}")

        try:
            filename, ret = _unwrap_ret(self._model.GetModelFilename())
            if ret == 0 and filename:
                self.meta.model_filename = str(filename)
        except Exception as e:
            self.meta.add_warning(f"No se pudo leer nombre del modelo: {e}")

    @property
    def model(self) -> Any:
        if self._model is None:
            raise RuntimeError("No conectado a SAP2000. Llama connect() primero.")
        return self._model

    @property
    def connected(self) -> bool:
        return self._model is not None


# ═══════════════════════════════════════════════
#  ANGLE READER
# ═══════════════════════════════════════════════


class AngleReader:
    """Lee datos de elementos angulares desde SAP2000."""

    def __init__(self, model: Any, warnings: list[str] | None = None):
        self._model = model
        self.warnings: list[str] = warnings if warnings is not None else []

    def _warn(self, message: str) -> None:
        self.warnings.append(message)

    def _check(self, ret: int, operation: str) -> bool:
        """Registra advertencia si el código de retorno indica error.

        Retorna True si la operación fue exitosa (ret == 0).
        """
        if ret != 0:
            self._warn(f"{operation}: SAP2000 retornó código {ret}")
            return False
        return True

    def list_combos(self) -> list[str]:
        """Lista las combinaciones de respuesta (RespCombo) disponibles."""
        try:
            _, names, ret = self._model.RespCombo.GetNameList()
            if not self._check(ret, "RespCombo.GetNameList"):
                return []
            return [n for n in names if n]
        except Exception as e:
            self._warn(f"RespCombo.GetNameList: {e}")
            return []

    def list_cases(self) -> list[str]:
        """Lista los casos de carga (LoadCases) disponibles."""
        try:
            _, names, ret = self._model.LoadCases.GetNameList()
            if not self._check(ret, "LoadCases.GetNameList"):
                return []
            return [n for n in names if n]
        except Exception as e:
            self._warn(f"LoadCases.GetNameList: {e}")
            return []

    def get_selected_frames(self) -> list[str]:
        """Retorna lista de nombres de frames seleccionados."""
        try:
            _, names, _, ret = self._model.SelectObj.GetSelected()
            if not self._check(ret, "SelectObj.GetSelected"):
                return []
            # Filtrar solo frames
            frame_names = []
            for name in names:
                if name and name.strip():
                    # Verificar que sea un frame
                    try:
                        _, prop, _, _ = self._model.FrameObj.GetSection(name)
                        if prop:
                            frame_names.append(name)
                    except Exception:
                        continue
            return frame_names
        except Exception as e:
            self._warn(f"SelectObj.GetSelected: {e}")
            return []

    def get_section_props(self, section_name: str) -> dict | None:
        """Obtiene propiedades de una sección de catálogo por nombre."""
        # Buscar en el catálogo interno (SQLite)
        try:
            from steeldesigner.catalog.catalog import Catalog
            from steeldesigner.core.section_adapter import to_angle_args
            cat = Catalog.shared()
            results = cat.fts_search(section_name, limit=5)
            for sec in results:
                fam = sec.family.family_code if sec.family else ""
                if fam in ("L_ICHA_LAM", "L_ICHA_PLEG", "L_AISC", "L"):
                    args = to_angle_args(sec)
                    return {
                        "b1": args["b1"], "b2": args["b2"], "t": args["t"],
                        "designation": sec.designation_modern or sec.designation_legacy or section_name,
                    }
        except Exception:
            pass

        # Leer propiedades desde SAP2000 directamente
        return self._read_section_from_sap(section_name)

    def _read_section_from_sap(self, section_name: str) -> dict | None:
        """Lee propiedades de sección desde SAP2000 vía OAPI."""
        try:
            _, prop = self._model.PropFrame.GetNameList()
            if section_name not in prop:
                return None

            # Para secciones Angulares (importadas o catalog)
            _, s_type, _, ret = self._model.PropFrame.GetSection(section_name)
            if not self._check(ret, f"PropFrame.GetSection({section_name})"):
                return None

            _, mat_prop, ret = self._model.PropFrame.GetMaterial(section_name)
            if not self._check(ret, f"PropFrame.GetMaterial({section_name})"):
                return None

            # Leer dimensiones del catálogo si la sección tiene nombre tipo "L100x100x10"
            import re
            m = re.match(r"L?(\d+)x(\d+)x(\d+)", section_name, re.I)
            if m:
                b1, b2, t = float(m.group(1)), float(m.group(2)), float(m.group(3))
                # Asegurar b1 >= b2
                if b1 < b2:
                    b1, b2 = b2, b1
                return {"b1": b1, "b2": b2, "t": t, "designation": section_name}

            self._warn(f"No se pudo determinar dimensiones de la sección '{section_name}'.")
            return None
        except Exception as e:
            self._warn(f"Error leyendo sección '{section_name}' desde SAP2000: {e}")
            return None

    def get_element_length(self, frame_name: str) -> float:
        """Obtiene el largo de un frame (mm, con unidades de trabajo fijadas)."""
        try:
            _, j1, j2, _, ret = self._model.FrameObj.GetPoints(frame_name, "", "")
            if not self._check(ret, f"FrameObj.GetPoints({frame_name})"):
                return 0.0

            _, x1, y1, z1, ret1 = self._model.PointObj.GetCoordCartesian(j1)
            _, x2, y2, z2, ret2 = self._model.PointObj.GetCoordCartesian(j2)
            if not self._check(ret1, f"PointObj.GetCoordCartesian({j1})") or \
               not self._check(ret2, f"PointObj.GetCoordCartesian({j2})"):
                return 0.0

            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1
            return math.sqrt(dx**2 + dy**2 + dz**2)
        except Exception as e:
            self._warn(f"Error midiendo largo de '{frame_name}': {e}")
            return 0.0

    def get_frame_forces(
        self, frame_name: str, selected: list[dict] | None = None
    ) -> tuple[float, float, str]:
        """Obtiene máxima compresión y tracción (kN) para el frame dado.

        Args:
            frame_name: nombre del frame.
            selected: lista de `{"name": str, "kind": "combo"|"case"}` con las
                combinaciones/casos a usar como salida. Si es None o vacía,
                se advierte (regla: nunca leer sin seleccionar salida
                explícitamente) y se usa el comportamiento de compatibilidad
                (todos los casos, sin distinguir cuál es crítico).

        Returns:
            (max_compression_kN, max_tension_kN, critical_combo_name)
        """
        if not selected:
            self._warn(
                f"'{frame_name}': no se seleccionó combinación/caso de salida; "
                "se leyeron todos los casos disponibles (no recomendado)."
            )
            return self._get_frame_forces_all_cases(frame_name)

        max_comp = 0.0
        max_tens = 0.0
        critical = ""
        for sel in selected:
            name = sel.get("name", "")
            kind = sel.get("kind", "combo")
            if not name:
                continue
            try:
                ret = _setter_ret(self._model.Results.Setup.DeselectAllCasesAndCombosForOutput())
                self._check(ret, "Results.Setup.DeselectAllCasesAndCombosForOutput")

                if kind == "combo":
                    ret = _setter_ret(self._model.Results.Setup.SetComboSelectedForOutput(name))
                else:
                    ret = _setter_ret(self._model.Results.Setup.SetCaseSelectedForOutput(name))
                if not self._check(ret, f"Results.Setup select '{name}'"):
                    continue

                result = self._model.Results.FrameForce(frame_name, 0)
                p = result[6]
                ret = result[-1]
                if not self._check(ret, f"Results.FrameForce({frame_name}, {name})") or not p:
                    continue

                comp = abs(min(p)) if min(p) < 0 else 0.0
                tens = max(p) if max(p) > 0 else 0.0
                if comp > max_comp:
                    max_comp = comp
                    critical = name
                max_tens = max(max_tens, tens)
            except Exception as e:
                self._warn(f"Error leyendo fuerzas de '{frame_name}' para '{name}': {e}")
                continue

        return (max_comp, max_tens, critical)

    def _get_frame_forces_all_cases(self, frame_name: str) -> tuple[float, float, str]:
        """Comportamiento de compatibilidad: todos los casos, sin seleccionar."""
        try:
            ret = _setter_ret(self._model.Results.Setup.DeselectAllCasesAndCombosForOutput())
            self._check(ret, "Results.Setup.DeselectAllCasesAndCombosForOutput")
            result = self._model.Results.FrameForce(frame_name, 0)
            p = result[6]
            ret = result[-1]
            if not self._check(ret, f"Results.FrameForce({frame_name})") or not p:
                return (0.0, 0.0, "")

            max_comp = abs(min(p)) if min(p) < 0 else 0.0
            max_tens = max(p) if max(p) > 0 else 0.0
            return (max_comp, max_tens, "")
        except Exception as e:
            self._warn(f"Error leyendo fuerzas de '{frame_name}': {e}")
            return (0.0, 0.0, "")

    def get_current_K(self, frame_name: str) -> tuple[float, float]:
        """Obtiene factores K actuales del frame.

        Returns:
            (K_strong, K_weak)
        """
        try:
            _, modifiers, ret = self._model.FrameObj.GetModifiers(frame_name)
            if not self._check(ret, f"FrameObj.GetModifiers({frame_name})"):
                return (1.0, 1.0)

            # Índices típicos en el array de modifiers:
            # [0]=Area, [1]=I33, [2]=I22, [3]=Mass, [4]=Weight,
            # [5]=Torsion, [6]=KFactorStrong, [7]=KFactorWeak
            k_strong = modifiers[6] if len(modifiers) > 6 else 1.0
            k_weak = modifiers[7] if len(modifiers) > 7 else 1.0

            return (float(k_strong), float(k_weak))
        except Exception as e:
            self._warn(f"Error leyendo K de '{frame_name}': {e}")
            return (1.0, 1.0)

    def get_material_props(self, section_name: str) -> tuple[float, float]:
        """Obtiene Fy y E del material de la sección.

        Returns:
            (Fy_MPa, E_MPa) — defaults 250, 200000 si no se puede leer
        """
        try:
            _, mat_name, ret = self._model.PropFrame.GetMaterial(section_name)
            if not self._check(ret, f"PropFrame.GetMaterial({section_name})"):
                return (250.0, 200000.0)

            _, _, _, Fy, Fu, _, ret = self._model.PropMaterial.GetOStress(mat_name, 0)
            if not self._check(ret, f"PropMaterial.GetOStress({mat_name})"):
                _, E, _, _, _, _, _ = self._model.PropMaterial.GetMPIsotropic(mat_name, 0)
                return (250.0, float(E) if E else 200000.0)

            _, E, _, _, _, _, _ = self._model.PropMaterial.GetMPIsotropic(mat_name, 0)
            return (float(Fy), float(E) if E else 200000.0)
        except Exception as e:
            self._warn(f"Error leyendo material de '{section_name}': {e}")
            return (250.0, 200000.0)

    def get_frame_section(self, frame_name: str) -> str:
        """Obtiene el nombre de la sección asignada a un frame."""
        try:
            _, section, _, ret = self._model.FrameObj.GetSection(frame_name)
            if not self._check(ret, f"FrameObj.GetSection({frame_name})"):
                return ""
            return str(section)
        except Exception as e:
            self._warn(f"Error leyendo sección de '{frame_name}': {e}")
            return ""

    def get_conn_leg(self, frame_name: str, override: str | None = None) -> tuple[str, bool]:
        """Determina qué pierna está conectada.

        Args:
            override: si se entrega "long"/"short", se usa directamente
                (decisión manual del usuario) y no se marca como default.

        Returns:
            (conn_leg, is_default) — is_default=True indica que no se pudo
            determinar geométricamente y se usó "long" como supuesto; la UI
            debe mostrar una advertencia visible en ese caso.

        Nota: SAP2000 no expone directamente "qué pierna quedó conectada"; se
        intenta leer la orientación de diseño (`GetDesignOrientation`) como
        indicio, pero la determinación geométrica completa requiere conocer
        la excentricidad real de la conexión modelada, que no siempre está
        disponible en el objeto de análisis. Por eso se prioriza el override
        manual del usuario.
        """
        if override in ("long", "short"):
            return override, False

        try:
            _, angle_deg, ret = self._model.FrameObj.GetDesignOrientation(frame_name)
            if ret == 0 and angle_deg is not None:
                # Heurística conservadora: sin geometría de excentricidad
                # confiable, no se infiere automáticamente la pierna. Se dev
                # welve el default y se advierte explícitamente.
                pass
        except Exception:
            pass

        self._warn(
            f"'{frame_name}': no se pudo determinar la pierna conectada; "
            "se asumió pierna larga por defecto. Verifica/corrige manualmente."
        )
        return "long", True

    def read_element(
        self,
        frame_name: str,
        selected_combos: list[dict] | None = None,
        conn_leg_override: str | None = None,
    ) -> AngleElementData | None:
        """Lee todos los datos de un elemento angular desde SAP2000."""
        try:
            section = self.get_frame_section(frame_name)
            if not section:
                return None

            sec_props = self.get_section_props(section)
            if sec_props is None:
                return None

            length = self.get_element_length(frame_name)
            if length <= 0:
                return None

            comp, tens, critical = self.get_frame_forces(frame_name, selected_combos)
            k_s, k_w = self.get_current_K(frame_name)
            fy, e = self.get_material_props(section)
            conn, conn_default = self.get_conn_leg(frame_name, conn_leg_override)

            return AngleElementData(
                name=frame_name,
                section_name=section,
                b1=sec_props["b1"],
                b2=sec_props["b2"],
                t=sec_props["t"],
                length_mm=length,
                conn_leg=conn,
                conn_leg_is_default=conn_default,
                Pu_comp_max=comp,
                Pu_tens_max=tens,
                critical_combo=critical,
                K_strong_current=k_s,
                K_weak_current=k_w,
                Fy_MPa=fy,
                E_MPa=e,
            )
        except Exception as e:
            self._warn(f"Error leyendo elemento {frame_name}: {e}")
            print(f"Error leyendo elemento {frame_name}: {e}", file=sys.stderr)
            return None


# ═══════════════════════════════════════════════
#  ANGLE WRITER
# ═══════════════════════════════════════════════


class AngleWriter:
    """Escribe modificaciones en SAP2000."""

    def __init__(self, model: Any, warnings: list[str] | None = None):
        self._model = model
        self.warnings: list[str] = warnings if warnings is not None else []

    def set_K_factor(self, frame_name: str, K_strong: float, K_weak: float) -> bool:
        """Modifica el factor K de un frame.

        Args:
            frame_name: nombre del frame
            K_strong: nuevo K para eje fuerte
            K_weak: nuevo K para eje débil

        Returns:
            True si se modificó correctamente
        """
        try:
            # Leer modifiers actuales
            _, modifiers, ret = self._model.FrameObj.GetModifiers(frame_name)
            if ret != 0:
                self.warnings.append(f"FrameObj.GetModifiers({frame_name}): código {ret}")
                return False

            # Asegurar que sea mutable (tuple → list)
            modifiers = list(modifiers)

            # Asegurar que el array tenga al menos 8 elementos
            while len(modifiers) < 8:
                modifiers.append(1.0)

            modifiers[6] = K_strong
            modifiers[7] = K_weak

            ret = self._model.FrameObj.SetModifiers(frame_name, modifiers)
            if ret != 0:
                self.warnings.append(f"FrameObj.SetModifiers({frame_name}): código {ret}")
                return False
            return True
        except Exception as e:
            self.warnings.append(f"Error modificando K de {frame_name}: {e}")
            print(f"Error modificando K de {frame_name}: {e}", file=sys.stderr)
            return False

    def set_KLr_as_overwrite(self, frame_name: str, KLr_eff: float) -> bool:
        """Establece el (KL/r) efectivo como overwrite de diseño.

        La firma real de la OAPI es `DesignSteel.SetOverwrite(Name, Item,
        Value, ItemType)` (Name es el string del frame, Item es un código
        entero de la tabla de overwrites del código de diseño activo, Value
        es el valor y ItemType es opcional con 0=Object por defecto). La
        implementación anterior invertía el orden (pasaba ItemType donde iba
        Name, y un string donde iba el Item entero), por lo que la escritura
        nunca llegaba a aplicarse contra un SAP2000 real.

        No se implementa aquí porque el código entero exacto de "Item" para
        "(KL/r) efectivo"/"Design KLr" varía según la tabla de overwrites del
        código de acero activo en el modelo (AISC 360-16, AISC 360-22, etc.)
        y no puede confirmarse sin la documentación OAPI de esa versión
        específica; adivinar el número arriesgaría escribir sobre otro
        parámetro de diseño sin que el usuario lo note. El factor K sí se
        escribe de forma verificada vía `set_K_factor()` (FrameObj.SetModifiers,
        API documentada y estable) — este método queda deshabilitado hasta
        confirmar el código de Item correcto contra la documentación CSI
        OAPI de la versión de SAP2000 en uso.
        """
        self.warnings.append(
            f"'{frame_name}': overwrite de (KL/r) de diseño no aplicado — "
            "el código de Item de SetOverwrite no está confirmado para el "
            "código de acero activo; usa el factor K escrito por FrameObj.SetModifiers."
        )
        return False


# ═══════════════════════════════════════════════
#  ANGLE SERVICE (orquestador)
# ═══════════════════════════════════════════════


class AngleService:
    """Orquesta: leer → calcular K → escribir en SAP2000."""

    def __init__(self, connector: Sap2000Connector, method: str = "LRFD"):
        self._connector = connector
        self.warnings: list[str] = []
        self._reader = AngleReader(connector.model, warnings=self.warnings)
        self._writer = AngleWriter(connector.model, warnings=self.warnings)
        self._method = method
        self._results: list[KFactorResult] = []

    def process_selection(
        self,
        selected_combos: list[dict] | None = None,
        conn_leg_override: str | None = None,
    ) -> list[KFactorResult]:
        """Procesa todos los frames seleccionados.

        1. Lee datos de cada frame (fuerzas del combo/caso seleccionado)
        2. Calcula nuevo K según AISC §E5
        3. Modifica K en SAP2000

        Returns:
            Lista de resultados por elemento
        """
        self._results = []
        frames = self._reader.get_selected_frames()

        if not frames:
            print("No hay frames seleccionados en SAP2000.", file=sys.stderr)
            return []

        for fname in frames:
            result = self._process_one(fname, selected_combos, conn_leg_override)
            if result:
                self._results.append(result)

        return self._results

    def _process_one(
        self,
        frame_name: str,
        selected_combos: list[dict] | None,
        conn_leg_override: str | None,
    ) -> KFactorResult | None:
        """Procesa un solo frame."""
        data = self._reader.read_element(frame_name, selected_combos, conn_leg_override)
        if data is None:
            return None

        from steeldesigner.core.angle_compression import check_angle

        # Compresión §E5 usando motor interno
        b1, b2 = (data.b1, data.b2) if data.b1 >= data.b2 else (data.b2, data.b1)
        comp = check_angle(
            b1=b1, b2=b2, t=data.t,
            L=data.length_mm,
            Pu=data.Pu_comp_max,  # ya en kN
            Fy=data.Fy_MPa, E=data.E_MPa,
            conn_leg=data.conn_leg,
            method=self._method,
        )

        phiPn_kN = comp.get("phiPn", 0.0)
        KLr_eff = comp.get("KLr_eff", 0.0)
        KLr_design = comp.get("KLr_design", KLr_eff)
        K_calc = KLr_eff / (data.length_mm / comp.get("r", 1.0)) if comp.get("r", 0) > 0 else 1.0

        ratio_comp = data.Pu_comp_max / phiPn_kN if phiPn_kN > 0 else None
        ratio_tens = data.Pu_tens_max / phiPn_kN if phiPn_kN > 0 else None
        ok_comp = ratio_comp <= 1.0 if ratio_comp is not None else None
        ok_tens = ratio_tens <= 1.0 if ratio_tens is not None else None

        # Escribir nuevo K en SAP2000
        # K_strong = K del eje paralelo a la pierna conectada
        # K_weak = K del otro eje
        if data.conn_leg == "long":
            k_strong_new = K_calc
            k_weak_new = data.K_weak_current  # mantener
        else:
            k_strong_new = data.K_strong_current  # mantener
            k_weak_new = K_calc

        written = self._writer.set_K_factor(frame_name, k_strong_new, k_weak_new)

        calc_steps = comp.get("calc_steps", [])

        return KFactorResult(
            element=frame_name,
            section=data.section_name,
            L_mm=data.length_mm,
            conn_leg=data.conn_leg,
            KLr_eff=round(KLr_eff, 2),
            K_calculated=round(K_calc, 4),
            K_strong_old=round(data.K_strong_current, 4),
            K_weak_old=round(data.K_weak_current, 4),
            K_strong_new=round(k_strong_new, 4),
            K_weak_new=round(k_weak_new, 4),
            phiPn_kN=round(phiPn_kN, 2),
            Pu_comp_max=round(data.Pu_comp_max, 2),
            Pu_tens_max=round(data.Pu_tens_max, 2),
            ratio_comp=round(ratio_comp, 3) if ratio_comp else None,
            ratio_tens=round(ratio_tens, 3) if ratio_tens else None,
            ok_comp=ok_comp,
            ok_tens=ok_tens,
            critical_combo=data.critical_combo,
            calc_steps=calc_steps,
            b1=data.b1,
            b2=data.b2,
            t=data.t,
            conn_leg_is_default=data.conn_leg_is_default,
            method=self._method,
        )

    def summary(self) -> str:
        """Genera resumen de resultados."""
        if not self._results:
            return "No se procesaron elementos."

        lines = ["=" * 64]
        lines.append("  AJUSTE DE FACTOR K — ÁNGULOS AISC §E5")
        lines.append("=" * 64)

        for r in self._results:
            lines.append(f"\n  {r.element}  |  {r.section}  |  L={r.L_mm/1000:.2f}m")
            lines.append(f"  {'─' * 56}")
            lines.append(f"  Conexión:     pierna {r.conn_leg}")
            lines.append(f"  K antiguo:    strong={r.K_strong_old}  weak={r.K_weak_old}")
            lines.append(f"  K nuevo:      strong={r.K_strong_new}  weak={r.K_weak_new}")
            lines.append(f"  (KL/r)ef:     {r.KLr_eff}")
            lines.append(f"  φcPn:         {r.phiPn_kN} kN")
            lines.append(f"  Compresión:   {r.Pu_comp_max} kN  {'✓ OK' if r.ok_comp else '✗ FALLA'}  ({r.ratio_comp})")
            lines.append(f"  Tracción:     {r.Pu_tens_max} kN  {'✓ OK' if r.ok_tens else '—'}")
            if r.critical_combo:
                lines.append(f"  Combo crítica: {r.critical_combo}")

        lines.append("\n" + "=" * 64)
        return "\n".join(lines)


# ═══════════════════════════════════════════════
#  FAKE SAP2000 (para testing sin SAP2000)
# ═══════════════════════════════════════════════


class _FakeFrameObj:
    """Frame SAP2000 falsa para testing."""

    def __init__(self):
        self.sections: dict[str, str] = {
            "F1": "L100x100x10",
            "F2": "L150x90x10",
        }
        self.coords: dict[str, tuple] = {
            "J1": (0, 0, 0),
            "J2": (0, 0, 3000),
            "J3": (5000, 0, 0),
            "J4": (5000, 0, 3000),
        }
        self.frame_joints: dict[str, tuple[str, str]] = {
            "F1": ("J1", "J2"),
            "F2": ("J3", "J4"),
        }
        self.modifiers: dict[str, list[float]] = {
            "F1": [1.0]*12,
            "F2": [1.0]*12,
        }
        # Índice 6=KStrong, 7=KWeak
        self.modifiers["F1"][6] = 1.0
        self.modifiers["F1"][7] = 1.0
        self.modifiers["F2"][6] = 1.0
        self.modifiers["F2"][7] = 1.0

    def GetSection(self, name):
        sec = self.sections.get(name, "")
        return (0, sec, 0, 0)

    def GetPoints(self, name, *args):
        j1, j2 = self.frame_joints.get(name, ("", ""))
        return (0, j1, j2, 0, 0)

    def GetModifiers(self, name):
        mods = self.modifiers.get(name, [1.0]*12)
        return (0, tuple(mods), 0)

    def SetModifiers(self, name, modifiers):
        self.modifiers[name] = list(modifiers)
        return 0

    def GetDesignOrientation(self, name):
        return (0, 0.0, 0)


class _FakePointObj:
    def GetCoordCartesian(self, name):
        c = {
            "J1": (0.0, 0.0, 0.0, 0),
            "J2": (0.0, 0.0, 3000.0, 0),
            "J3": (5000.0, 0.0, 0.0, 0),
            "J4": (5000.0, 0.0, 3000.0, 0),
        }
        x, y, z, _ = c.get(name, (0, 0, 0, 0))
        return (0, x, y, z, 0)


class _FakeSelectObj:
    def GetSelected(self):
        return (0, ["F1", "F2"], ["Frame", "Frame"], 0)


class _FakeResultsSetup:
    """Registra las llamadas de selección de salida para poder testearlas."""

    def __init__(self):
        self.selected: list[tuple[str, str]] = []
        self.calls: list[str] = []

    def DeselectAllCasesAndCombosForOutput(self):
        # Retorna un int suelto (el código de retorno), como el resto de las
        # funciones "Set*" de la OAPI real — NO una tupla (valor, ret).
        self.selected = []
        self.calls.append("deselect_all")
        return 0

    def SetComboSelectedForOutput(self, name):
        self.selected.append((name, "combo"))
        self.calls.append(f"select_combo:{name}")
        return 0

    def SetCaseSelectedForOutput(self, name):
        self.selected.append((name, "case"))
        self.calls.append(f"select_case:{name}")
        return 0


class _FakeResults:
    def __init__(self):
        self.Setup = _FakeResultsSetup()

    def FrameForce(self, name, *args):
        # Los valores varían levemente según el combo/caso seleccionado,
        # para poder testear que la selección realmente afecta el resultado.
        selected_names = {n for n, _ in self.Setup.selected}
        base = {
            "F1": (2, ["F1", "F1"], [1, 1], ["DEAD", "DEAD"],
                   ["StepNum", "StepNum"], [0, 0],
                   [-80.0, 30.0], [0, 0], [0, 0],
                   [0, 0], [0, 0], [0, 0], 0),
            "F2": (2, ["F2", "F2"], [1, 1], ["DEAD", "DEAD"],
                   ["StepNum", "StepNum"], [0, 0],
                   [-120.0, 45.0], [0, 0], [0, 0],
                   [0, 0], [0, 0], [0, 0], 0),
        }
        row = base.get(name, (0,))
        if len(row) < 7:
            return row
        if "COMB_ENVOLVENTE" in selected_names:
            # combinación crítica con mayor demanda
            p = tuple(v * 1.5 for v in row[6])
            row = row[:6] + (p,) + row[7:]
        return row


class _FakePropFrame:
    def GetNameList(self):
        return (2, ["L100x100x10", "L150x90x10"], 0)

    def GetMaterial(self, name):
        return (0, "A36", 0)

    def GetSection(self, name):
        return (0, 5, "Angle", 0)


class _FakePropMaterial:
    def GetOStress(self, name, *args):
        return (0, 0, 0, 250.0, 400.0, 0.25, 0)

    def GetMPIsotropic(self, name, *args):
        return (0, 200000.0, 0.3, 1.17e-5, 0)


class _FakeDesignSteel:
    def SetOverwrite(self, item_type, name, value):
        return 0


class _FakeRespCombo:
    def GetNameList(self):
        return (2, ["COMB1", "COMB_ENVOLVENTE"], 0)


class _FakeLoadCases:
    def GetNameList(self):
        return (2, ["DEAD", "LIVE"], 0)


class FakeSapModel:
    """Modelo SAP2000 falso para testing."""

    def __init__(self):
        self.FrameObj = _FakeFrameObj()
        self.PointObj = _FakePointObj()
        self.SelectObj = _FakeSelectObj()
        self.Results = _FakeResults()
        self.PropFrame = _FakePropFrame()
        self.PropMaterial = _FakePropMaterial()
        self.DesignSteel = _FakeDesignSteel()
        self.RespCombo = _FakeRespCombo()
        self.LoadCases = _FakeLoadCases()
        self._units = 6  # supuesto: kgf_m_C antes de conectar

    def GetPresentUnits(self):
        return (self._units, 0)

    def SetPresentUnits(self, units):
        self._units = units
        return 0

    def GetModelFilename(self):
        return ("FakeModel.sdb", 0)


class FakeSapObject:
    """SapObject falso (raíz), expone GetVersion()."""

    def __init__(self):
        self.SapModel = FakeSapModel()

    def GetVersion(self):
        return ("23.0.0", 0)
