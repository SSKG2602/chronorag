from __future__ import annotations

from typing import Dict


def report_incident(payload: Dict) -> Dict:
    return {"status": "received", "payload": payload}
