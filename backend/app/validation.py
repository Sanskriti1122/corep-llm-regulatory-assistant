from __future__ import annotations

from typing import Any, Dict, List, Optional

from .models import CorepResult


NUMERIC_FIELDS = ["CET1", "AT1", "Tier2", "RWA", "CET1_ratio"]
MIN_CET1_RATIO = 0.045


def _to_optional_float(value: Any):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip().replace("%", "")
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def validate_corep_result(initial: Dict[str, Any]):
    template_name = str(initial.get("template_name") or "COREP Own Funds (prototype)")
    explanation = str(
        initial.get(
            "explanation",
            "Explanation not provided by the model. This is an automatically generated prototype output.",
        )
    )

    cet1 = _to_optional_float(initial.get("CET1"))
    at1 = _to_optional_float(initial.get("AT1"))
    tier2 = _to_optional_float(initial.get("Tier2"))
    rwa = _to_optional_float(initial.get("RWA"))
    cet1_ratio_model = _to_optional_float(initial.get("CET1_ratio"))

    missing_fields = []
    for f_name, value in [("CET1", cet1), ("AT1", at1), ("Tier2", tier2), ("RWA", rwa)]:
        if value is None:
            missing_fields.append(f_name)

    validation_warnings = []

    cet1_ratio = None
    if cet1 is not None and rwa is not None and rwa > 0:
        cet1_ratio = cet1 / rwa
        if cet1_ratio_model is not None:
            if abs(cet1_ratio_model - cet1) > 1e-6 and abs(cet1_ratio_model - cet1_ratio) > 1e-6:
                validation_warnings.append(
                    "Model-proposed CET1_ratio differed from deterministic CET1/RWA. "
                    "The deterministic value has been used instead."
                )
    else:
        cet1_ratio = cet1_ratio_model

    if cet1_ratio is not None and cet1_ratio < MIN_CET1_RATIO:
        validation_warnings.append(
            f"CET1_ratio of {cet1_ratio:.4f} is below the minimum requirement of "
            f"{MIN_CET1_RATIO:.4f} (4.5%)."
        )

    rules_used_raw = initial.get("rules_used") or []
    rules_used = []
    if isinstance(rules_used_raw, list):
        for item in rules_used_raw:
            try:
                rules_used.append(str(item))
            except:
                continue
    else:
        try:
            rules_used.append(str(rules_used_raw))
        except:
            pass

    model_missing = initial.get("missing_fields") or []
    if isinstance(model_missing, list):
        for entry in model_missing:
            entry_str = str(entry)
            if entry_str not in missing_fields:
                missing_fields.append(entry_str)

    model_warnings = initial.get("validation_warnings") or []
    if isinstance(model_warnings, list):
        for warning in model_warnings:
            warning_str = str(warning)
            if warning_str not in validation_warnings:
                validation_warnings.append(warning_str)

    result = CorepResult(
        template_name=template_name,
        CET1=cet1,
        AT1=at1,
        Tier2=tier2,
        RWA=rwa,
        CET1_ratio=cet1_ratio,
        missing_fields=missing_fields,
        validation_warnings=validation_warnings,
        rules_used=rules_used,
        explanation=explanation,
    )

    return result
