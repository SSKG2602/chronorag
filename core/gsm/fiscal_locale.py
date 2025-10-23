from __future__ import annotations

from typing import Dict


def infer_fiscal_locale(tenant_cfg: Dict) -> Dict:
    return {
        "fy_start": tenant_cfg.get("fy_start", "APRIL"),
        "locale": tenant_cfg.get("locale", "US"),
        "resolved": True,
    }
