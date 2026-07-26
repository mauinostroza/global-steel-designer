"""Lee fuerzas de elementos barra seleccionados en SAP2000 OAPI.

Reutiliza Sap2000Connector de sap2000_oapi.py.
Unidades de trabajo: kN, mm (WORK_UNITS_CODE=5). Se convierten a N y N·mm.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from steeldesigner.sap2000.sap2000_oapi import Sap2000Connector, WORK_UNITS_CODE


@dataclass(slots=True)
class FrameLoads:
    """Fuerzas de diseño de un elemento barra (unidades SI: N, N·mm)."""
    element_name: str
    combo_name: str          # nombre del combo/caso que controla
    Pu_N: float = 0.0        # axial (positivo = compresión convenio AISC)
    Mux_Nmm: float = 0.0     # momento eje fuerte
    Muy_Nmm: float = 0.0     # momento eje débil
    Vux_N: float = 0.0       # cortante eje fuerte
    length_mm: float = 0.0   # longitud del elemento


def read_selected_frame_loads(
    connector: Sap2000Connector,
    combo_or_case: str | None = None,
) -> list[FrameLoads]:
    """Lee fuerzas de los frames seleccionados en SAP2000.

    Args:
        connector: Sap2000Connector ya conectado
        combo_or_case: nombre de combo/caso a leer; None = envelope de todos

    Returns:
        Lista de FrameLoads, uno por elemento seleccionado. Lista vacía si
        no hay elementos seleccionados o no hay resultados disponibles.
    """
    model = connector.SapModel
    meta = connector.meta

    # Fijar unidades kN, mm
    ret = model.SetPresentUnits(WORK_UNITS_CODE)
    if ret != 0:
        meta.add_warning(f"SetPresentUnits retornó {ret}")

    # Elementos seleccionados
    num_frames, names, ret = model.FrameObj.GetNameList()
    if ret != 0 or num_frames == 0:
        return []

    # Filtrar solo los seleccionados
    selected_names: list[str] = []
    for name in names:
        is_sel, ret2 = model.FrameObj.GetSelected(name)
        if ret2 == 0 and is_sel:
            selected_names.append(name)

    if not selected_names:
        meta.add_warning("No hay frames seleccionados en SAP2000")
        return []

    # Determinar combos a leer
    if combo_or_case:
        combos_to_read = [combo_or_case]
    else:
        num_c, combo_names, ret3 = model.RespCombo.GetNameList()
        if ret3 == 0 and num_c > 0:
            combos_to_read = list(combo_names)
        else:
            num_lc, lc_names, ret4 = model.LoadCases.GetNameList()
            if ret4 == 0 and num_lc > 0:
                combos_to_read = list(lc_names)
            else:
                meta.add_warning("No se encontraron combos ni casos de carga")
                return []

    results: list[FrameLoads] = []

    for name in selected_names:
        # Longitud del elemento
        length_mm = _get_frame_length(model, name, meta)

        # Envelope de fuerzas sobre todos los combos
        env = _EnvelopeAccumulator()

        for combo in combos_to_read:
            # ItemTypeElm=2 → resultados por elemento (no por objeto)
            # NumberResults, Obj, ObjSta, Elm, ElmSta, LoadCase,
            # StepType, StepNum, P, V2, V3, T, M2, M3, ret
            out = model.Results.FrameForce(name, 2)
            # out es tupla variable — OAPI devuelve:
            # (NumberResults, Obj[], ObjSta[], Elm[], ElmSta[],
            #  LoadCase[], StepType[], StepNum[], P[], V2[], V3[],
            #  T[], M2[], M3[], ret)
            if len(out) < 15:
                continue
            n_res = out[0]
            if n_res == 0 or out[-1] != 0:
                continue

            load_cases = out[5]   # array de nombres de combo/caso
            P_arr = out[8]        # kN, positivo = tracción en SAP2000
            V2_arr = out[9]       # kN
            M3_arr = out[13]      # kN·mm, eje fuerte
            M2_arr = out[12]      # kN·mm, eje débil

            for i in range(n_res):
                if combo_or_case and load_cases[i] != combo_or_case:
                    continue
                # SAP2000 P positivo = tracción; AISC Pu compresión positiva
                Pu = -float(P_arr[i]) * 1000.0     # kN → N, signo invertido
                Mux = abs(float(M3_arr[i])) * 1000.0  # kN·mm → N·mm
                Muy = abs(float(M2_arr[i])) * 1000.0
                Vux = abs(float(V2_arr[i])) * 1000.0
                env.update(Pu, Mux, Muy, Vux, load_cases[i])

        if env.controlling_combo:
            results.append(FrameLoads(
                element_name=name,
                combo_name=env.controlling_combo,
                Pu_N=env.Pu,
                Mux_Nmm=env.Mux,
                Muy_Nmm=env.Muy,
                Vux_N=env.Vux,
                length_mm=length_mm,
            ))
        else:
            # Sin resultados para este elemento: incluir con ceros
            results.append(FrameLoads(
                element_name=name,
                combo_name="—",
                length_mm=length_mm,
            ))

    return results


# ─── helpers ──────────────────────────────────────────────────────────────────

def _get_frame_length(model, name: str, meta) -> float:
    """Retorna la longitud del frame en mm."""
    try:
        # GetPoints devuelve (point_i, point_j, ret)
        pt_i, pt_j, ret = model.FrameObj.GetPoints(name)
        if ret != 0:
            return 0.0
        xi, yi, zi, ret_i = model.PointObj.GetCoordCartesian(pt_i)
        xj, yj, zj, ret_j = model.PointObj.GetCoordCartesian(pt_j)
        if ret_i != 0 or ret_j != 0:
            return 0.0
        return math.sqrt((xj-xi)**2 + (yj-yi)**2 + (zj-zi)**2)
    except Exception as exc:
        meta.add_warning(f"No se pudo obtener largo de {name}: {exc}")
        return 0.0


class _EnvelopeAccumulator:
    """Acumula el envelope (máximo valor absoluto) de las fuerzas."""
    def __init__(self):
        self.Pu = 0.0
        self.Mux = 0.0
        self.Muy = 0.0
        self.Vux = 0.0
        self.controlling_combo: str | None = None
        self._max_demand = 0.0

    def update(self, Pu: float, Mux: float, Muy: float, Vux: float, combo: str):
        # Controla el combo con mayor demanda combinada (norma no-dimensional simple)
        demand = abs(Pu) + abs(Mux) / 1e6 + abs(Muy) / 1e6 + abs(Vux)
        if demand > self._max_demand:
            self._max_demand = demand
            self.controlling_combo = combo
        # Envelope individual por componente
        self.Pu = max(self.Pu, Pu, key=abs) if abs(Pu) > abs(self.Pu) else self.Pu
        self.Mux = max(self.Mux, Mux)
        self.Muy = max(self.Muy, Muy)
        self.Vux = max(self.Vux, Vux)
