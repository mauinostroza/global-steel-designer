"""Selector de perfiles — diseña un conjunto de secciones para unas cargas dadas
y las devuelve ordenadas por menor peso que cumple con D/C ≤ 1.0 (ranking).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

from steeldesigner.catalog.models import Section
from steeldesigner.core.engine_facade import DesignInputs, DesignResult, EngineFacade


@dataclass
class SelectorCandidate:
    section: Section
    result: DesignResult
    rank: int = 0


def select_profiles(
    sections: list[Section],
    inputs: DesignInputs,
    max_workers: int = 4,
    progress_cb: Callable[[int, int], None] | None = None,
) -> list[SelectorCandidate]:
    """Diseña todas las secciones y devuelve ranking ordenado.

    Orden:
      1. Grupo que PASA (interaction_ratio ≤ 1.0, sin error):
         - peso ASC, desempate por interaction_ratio DESC (mayor aprovechamiento)
      2. Grupo que NO PASA: interaction_ratio ASC

    Args:
        sections: lista de Section del catálogo
        inputs: cargas y parámetros de diseño
        max_workers: hilos paralelos (ThreadPoolExecutor)
        progress_cb: callback(completados, total) llamado después de cada cálculo
    """
    total = len(sections)
    completed = 0
    raw: list[tuple[Section, DesignResult]] = []

    facade = EngineFacade()

    def _run_one(sec: Section) -> tuple[Section, DesignResult]:
        return sec, facade.run(sec, inputs)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_one, s): s for s in sections}
        for fut in as_completed(futures):
            sec, res = fut.result()
            raw.append((sec, res))
            completed += 1
            if progress_cb:
                progress_cb(completed, total)

    passing = [
        (s, r) for s, r in raw
        if r.passes_interaction and r.error is None
    ]
    failing = [
        (s, r) for s, r in raw
        if not r.passes_interaction or r.error is not None
    ]

    passing.sort(key=lambda x: (x[0].weight_kg_m, -x[1].interaction_ratio))
    failing.sort(key=lambda x: x[1].interaction_ratio)

    ordered = passing + failing
    return [
        SelectorCandidate(section=s, result=r, rank=i + 1)
        for i, (s, r) in enumerate(ordered)
    ]
