from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class AuditItem:
    severity: str
    family: str
    chapter: str
    equation: str
    message: str

@dataclass
class AuditReport:
    items: List[AuditItem] = field(default_factory=list)
    @property
    def has_blockers(self) -> bool:
        return any(i.severity == "BLOCKER" for i in self.items)
    @property
    def has_warnings(self) -> bool:
        return any(i.severity == "WARNING" for i in self.items)
    def as_dicts(self) -> List[Dict[str, str]]:
        return [{"severity":i.severity,"family":i.family,"chapter":i.chapter,"equation":i.equation,"message":i.message} for i in self.items]

class PlaceholderAudit:
    WARNING_TOKENS = ["conservative", "approximate", "geometric_idealized", "practical"]
    BLOCK_TOKENS = ["blocked_exact_data", "blocking", "requires tabulated", "requires exact"]
    @staticmethod
    def inspect_limit_state(family: str, result: Any):
        items = []
        if result is None:
            return items
        chapter = getattr(result, "chapter", "?")
        equation = (getattr(result, "equation", "") or "").lower()
        desc = (getattr(result, "description", "") or "").lower()
        metadata = getattr(result, "metadata", {}) or {}
        text_pool = [equation, desc] + [str(v).lower() for v in metadata.values()]
        if any(token in text for token in PlaceholderAudit.BLOCK_TOKENS for text in text_pool):
            items.append(AuditItem("BLOCKER", family, chapter, getattr(result, "equation", ""), "Exact normative branch blocked due to missing required data."))
            return items
        if any(token in text for token in PlaceholderAudit.WARNING_TOKENS for text in text_pool):
            items.append(AuditItem("WARNING", family, chapter, getattr(result, "equation", ""), "Result uses geometric idealization or approximate/conservative logic."))
        source = str(metadata.get("source", "")).upper()
        if source in {"APPROXIMATE", "GEOMETRIC_IDEALIZED"}:
            items.append(AuditItem("WARNING", family, chapter, getattr(result, "equation", ""), f"Geometry source is {source}."))
        return items
    @staticmethod
    def inspect_bundle(family: str, bundle: Any):
        items = []
        if bundle is None:
            return items
        controlling = getattr(bundle, "controlling", None)
        if controlling is not None:
            items.extend(PlaceholderAudit.inspect_limit_state(family, controlling))
        notes = getattr(bundle, "notes", []) or []
        for note in notes:
            note_l = str(note).lower()
            if "not evaluated" in note_l or "requires" in note_l or "tension field action" in note_l:
                items.append(AuditItem("WARNING", family, getattr(controlling, "chapter", "?"), getattr(controlling, "equation", ""), str(note)))
        return items
    @staticmethod
    def inspect_result_package(family: str, result_package: Dict[str, Any]):
        report = AuditReport()
        for key, value in result_package.items():
            if key in {"interaction_ratio", "passes_interaction", "geometry_source", "audit_report"}:
                continue
            report.items.extend(PlaceholderAudit.inspect_bundle(family, value))
        geom = result_package.get("geometry_source")
        if isinstance(geom, dict):
            source = str(geom.get("source", "")).upper()
            if source in {"APPROXIMATE", "GEOMETRIC_IDEALIZED"}:
                report.items.append(AuditItem("WARNING", family, "GEOMETRY", source, f"Section geometry source is {source}."))
        return report

class StrictNormativeClosureError(RuntimeError):
    pass

class MasterEngineV2:
    def __init__(self, mode: str = "best_effort"):
        if mode not in {"best_effort", "strict"}:
            raise ValueError("mode must be 'best_effort' or 'strict'")
        self.mode = mode
    def _enforce(self, family: str, result_package: Dict[str, Any]):
        audit = PlaceholderAudit.inspect_result_package(family, result_package)
        result_package["audit_report"] = audit
        if self.mode == "strict" and audit.has_blockers:
            raise StrictNormativeClosureError(f"Family {family} contains blocked exact-data branches. Review audit_report.")
        return result_package
    def run_i_shape_member(self, member: Any, demand: Any): return self._enforce("I_SHAPE", member.check_all(demand))
    def run_channel_member(self, member: Any, demand: Any): return self._enforce("CHANNEL", member.check_all(demand))
    def run_angle_member(self, member: Any, demand: Any): return self._enforce("ANGLE", member.check_all(demand))
    def run_tee_member(self, member: Any, demand: Any): return self._enforce("TEE", member.check_all(demand))

@dataclass
class RequiredDataItem:
    family: str
    chapter: str
    description: str
    required_inputs: List[str]

def normative_gap_registry_v2():
    return [
        RequiredDataItem("CHANNEL","E/F","Exact singly symmetric channel closure in strict mode",["tabulated shear center","validated Cw","principal-axis properties"]),
        RequiredDataItem("ANGLE","E/F","Exact angle principal-axis and eccentricity closure in strict mode",["principal-axis properties","connection eccentricity model","tabulated or validated shear center"]),
        RequiredDataItem("TEE","E/F","Exact tee flexo-torsional and sign-dependent closure in strict mode",["principal-axis properties","tabulated or validated shear center","validated Cw"]),
        RequiredDataItem("ALL","Validation","Regression suite against benchmark examples",["reference examples","accepted tolerances","benchmark outputs"]),
    ]
