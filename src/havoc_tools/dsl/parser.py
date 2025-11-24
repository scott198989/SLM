from __future__ import annotations

import json
from typing import Any, Dict

import yaml

from havoc_tools.dsl.spec import DSLRequest, DesignDOE, EvalSPC, Factor


class DSLParseError(Exception):
    pass


def parse_dsl(content: str) -> DSLRequest:
    try:
        data: Dict[str, Any] = yaml.safe_load(content) if not content.strip().startswith("{") else json.loads(content)
    except Exception as exc:  # noqa: BLE001
        raise DSLParseError(f"Failed to parse DSL: {exc}") from exc

    request = DSLRequest(raw=data)
    if "DESIGN_DOE" in data:
        doe_data = data["DESIGN_DOE"]
        factors = [Factor(name=f["name"], levels=f["levels"]) for f in doe_data.get("factors", [])]
        request.design_doe = DesignDOE(
            type=doe_data.get("type", "UNKNOWN"),
            factors=factors,
            response=doe_data.get("response", ""),
            alpha=doe_data.get("alpha", 0.05),
        )
    if "EVAL_SPC" in data:
        spc_data = data["EVAL_SPC"]
        request.eval_spc = EvalSPC(
            chart=spc_data.get("chart", "XR"),
            subgroup_size=int(spc_data.get("subgroup_size", 5)),
            data_source=spc_data.get("data_source", ""),
            alpha=float(spc_data.get("alpha", 0.05)),
        )
    return request
