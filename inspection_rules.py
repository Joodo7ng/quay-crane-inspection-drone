"""Versioned workflow priorities and traceable, non-binding guidance."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

RANK = {"정상": 0, "경미": 1, "주의": 2, "위험": 3}
DEFAULT_CRITERIA = Path(__file__).parent / "criteria" / "inspection_v2.json"
OPERATORS = {
    "eq": lambda a, b: a == b,
    "in": lambda a, b: a in b,
    "gt": lambda a, b: a > b,
    "gte": lambda a, b: a >= b,
    "lt": lambda a, b: a < b,
    "lte": lambda a, b: a <= b,
}


def load_criteria(path=DEFAULT_CRITERIA):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if data.get("schema_version") != 1:
        raise ValueError("지원하지 않는 기준 데이터 형식임")
    ids = set()
    for rule in data["rules"]:
        if rule["id"] in ids or rule["priority"] not in RANK:
            raise ValueError("중복 규칙 또는 잘못된 우선순위임")
        ids.add(rule["id"])
        if any(c["operator"] not in OPERATORS for c in rule["conditions"]):
            raise ValueError("지원하지 않는 조건 연산임")
        if any(x not in data["references"] for x in rule["reference_ids"]):
            raise ValueError("근거 문서가 누락됨")
    data["sha256"] = hashlib.sha256(raw).hexdigest()
    return data


def normalize_context(context, criteria):
    c = deepcopy(context or {})
    c.setdefault("asset_id", "미지정")
    c.setdefault("inspection_id", "미지정")
    c.setdefault("component_type", "unknown")
    if c["component_type"] not in criteria["components"]:
        raise ValueError("등록되지 않은 부재임")
    comp = criteria["components"][c["component_type"]]
    c.setdefault("component_instance_id", "")
    c.setdefault("part", "미상")
    if c["part"] not in comp["parts"]:
        raise ValueError("해당 부재에 등록되지 않은 점검 부위임")
    c.setdefault("component_source", "manual")
    c["component_confirmed"] = (
        bool(c.get("component_confirmed")) and c["component_type"] != "unknown"
    )
    c.setdefault("quality", "unconfirmed")
    if c["quality"] not in ("usable", "unconfirmed", "poor"):
        raise ValueError("입력 품질 상태가 올바르지 않음")
    c["component_name"] = comp["name"]
    c["fcm_scope"] = comp["fcm"]
    c["is_connection"] = any(
        k in c["part"] for k in ("연결", "용접", "힌지", "지지", "로크 핀")
    )
    c["critical_context"] = c["component_confirmed"] and (
        comp["fcm"] == "yes"
        or (comp["fcm"] == "connections_only" and c["is_connection"])
        or (c["component_type"] == "boom_hinge" and c["part"] != "미상")
        or (
            c["component_type"] == "machinery"
            and c["part"] in ("트롤리 틀", "사재 연결부")
        )
    )
    return c


def matches(rule, facts):
    if rule["status"] != "active" or rule["review_status"] != "implementation_reviewed":
        return False
    # Only reviewed team workflow rules may assign a priority. External legal
    # references remain advisory until separately reviewed for applicability.
    if rule["category"] != "team_workflow":
        return False
    for cond in rule["conditions"]:
        value = facts.get(cond["field"])
        if value is None:
            return False
        try:
            if not OPERATORS[cond["operator"]](value, cond["value"]):
                return False
        except (TypeError, ValueError):
            return False
    return True


def judge_defect(defect, context, criteria):
    comp = criteria["components"][context["component_type"]]
    applied = [
        deepcopy(r) for r in criteria["rules"] if matches(r, {**context, **defect})
    ]
    priority = max((r["priority"] for r in applied), key=RANK.get, default=None)
    reasons = [r["reason"] for r in applied]
    actions = [a for r in applied for a in r["actions"]]
    checks = [a for r in applied for a in r["checks"]]
    refs = []
    if context["component_confirmed"]:
        ref_ids = list(comp["reference_ids"])
        if context["component_type"] == "leg" and not context["is_connection"]:
            ref_ids = [i for i in ref_ids if i != "G9-8"]
        if context["component_type"] == "leg" and context["is_connection"]:
            ref_ids.append("G9-8")
        refs = [deepcopy(criteria["references"][i]) for i in dict.fromkeys(ref_ids)]
        checks += comp["checks"]
        if context["part"] == "미상":
            checks.append("세부 점검 부위 확인이 필요함")
        reasons.append("부재 및 부위는 점검자가 지정한 정보이며 자동 인식 결과가 아님")
    else:
        checks.append("부재 및 후보 위치 확인이 필요함. 공통 지침만 제공함")
    if context["quality"] != "usable":
        checks.append("영상 품질 확인 또는 재촬영이 필요함")
    if not applied:
        reasons.append(
            "필수 측정값 또는 적용 가능한 검토 규칙이 없어 기준 적용을 보류함"
        )
    return {
        "priority": priority,
        "status": "applied" if applied else "deferred",
        "rule_ids": [r["id"] for r in applied],
        "rules": applied,
        "reasons": reasons,
        "actions": list(dict.fromkeys(actions)),
        "checks": list(dict.fromkeys(checks)),
        "references": refs,
        "legal_judgment": "미수행",
        "review_status": "점검자 확인 대기",
        "document_guidance": (
            document_guidance(defect, context, criteria)
            if "guidance_actions" in criteria
            else None
        ),
    }


def document_guidance(defect, context, criteria):
    """Attachment guidance, separate from automatic priority and legal judgment."""
    comp = criteria["components"][context["component_type"]]
    confirmed = context["component_confirmed"]
    ids = comp.get("judgment_reference_ids", []) if confirmed else []
    methods = comp.get("method_reference_ids", []) if confirmed else []
    if context["component_type"] == "leg" and not context["is_connection"]:
        methods = [i for i in methods if i != "G9-8"]
    kind = defect["defect"]
    action_key = (
        "unknown"
        if not confirmed
        else (
            "critical_crack"
            if kind == "균열" and context["critical_context"]
            else "crack" if kind == "균열" else "corrosion"
        )
    )
    action = criteria["guidance_actions"][action_key]
    if confirmed and context["component_type"] == "portal_beam" and kind == "균열":
        action += " 큰 균열로 확인되면 즉시 조치 필요성을 검토함(PEMA 3장). 영상만으로 크기를 확정하지 않음"
    return {
        "version": criteria["version"],
        "judgment_ids": ids,
        "method_ids": methods,
        "action": action,
        "reason": (
            f"{comp['name']} · {context['part']}의 {kind} 후보가 문서의 점검 대상에 해당함"
            if confirmed
            else "부재 미상·미확인으로 특정 부재 조항 적용을 보류함"
        ),
        "source": criteria["guidance_source"],
        "verification": "첨부 문서 기준 · 원문 재확인 미완료 · 최종 판단은 점검자 확인",
        "references": [
            deepcopy(criteria["references"][i]) for i in dict.fromkeys(ids + methods)
        ],
    }
