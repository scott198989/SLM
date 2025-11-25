from __future__ import annotations

import json
from typing import Any, Dict

import yaml

from havoc_tools.dsl.spec import (
    DSLRequest,
    DesignDOE,
    DOEOperation,
    EvalSPC,
    Factor,
    MathExpression,
    SPCOperation,
    StatTest,
)


class DSLParseError(Exception):
    pass


def parse_dsl(content: str) -> DSLRequest:
    try:
        data: Dict[str, Any] = yaml.safe_load(content) if not content.strip().startswith("{") else json.loads(content)
    except Exception as exc:  # noqa: BLE001
        raise DSLParseError(f"Failed to parse DSL: {exc}") from exc

    request = DSLRequest(raw=data)

    # Parse DESIGN_DOE (legacy format)
    if "DESIGN_DOE" in data:
        doe_data = data["DESIGN_DOE"]
        factors = [Factor(name=f["name"], levels=f["levels"]) for f in doe_data.get("factors", [])]
        request.design_doe = DesignDOE(
            type=doe_data.get("type", "UNKNOWN"),
            factors=factors,
            response=doe_data.get("response", ""),
            alpha=doe_data.get("alpha", 0.05),
            blocks=doe_data.get("blocks"),
            replicates=doe_data.get("replicates", 1),
        )

    # Parse EVAL_SPC (legacy format)
    if "EVAL_SPC" in data:
        spc_data = data["EVAL_SPC"]
        request.eval_spc = EvalSPC(
            chart=spc_data.get("chart", "XR"),
            subgroup_size=int(spc_data.get("subgroup_size", 5)),
            data_source=spc_data.get("data_source", ""),
            alpha=float(spc_data.get("alpha", 0.05)),
            rules=spc_data.get("rules", ["WECO_1", "WECO_2", "WECO_3", "WECO_4"]),
        )

    # Parse MATH operation
    if "MATH" in data:
        math_data = data["MATH"]
        request.math_expr = MathExpression(
            expression=math_data.get("expression", ""),
            variables=math_data.get("variables", {}),
            symbolic=math_data.get("symbolic", False),
        )

    # Parse STAT_TEST operation
    if "STAT_TEST" in data:
        stat_data = data["STAT_TEST"]
        request.stat_test = StatTest(
            test_type=stat_data.get("test_type", "ttest"),
            data_a=stat_data.get("data_a"),
            data_b=stat_data.get("data_b"),
            formula=stat_data.get("formula"),
            data_frame=stat_data.get("data_frame"),
            alpha=stat_data.get("alpha", 0.05),
            equal_var=stat_data.get("equal_var", False),
        )

    # Parse DOE_OPERATION
    if "DOE" in data:
        doe_data = data["DOE"]
        factors = [Factor(name=f["name"], levels=f["levels"]) for f in doe_data.get("factors", [])]
        request.doe_operation = DOEOperation(
            operation=doe_data.get("operation", "factorial"),
            factors=factors,
            response_data=doe_data.get("response_data"),
            center_points=doe_data.get("center_points", 0),
            blocks=doe_data.get("blocks"),
        )

    # Parse SPC_OPERATION
    if "SPC" in data:
        spc_data = data["SPC"]
        request.spc_operation = SPCOperation(
            chart_type=spc_data.get("chart_type", "XBar_R"),
            data=spc_data.get("data", []),
            subgroup_size=spc_data.get("subgroup_size"),
            rules=spc_data.get("rules", ["WECO_1"]),
            target=spc_data.get("target"),
            ucl=spc_data.get("ucl"),
            lcl=spc_data.get("lcl"),
        )

    return request
