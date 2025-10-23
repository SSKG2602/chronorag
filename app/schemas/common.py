from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class TimeAxis(str, Enum):
    valid = "valid"
    transaction = "transaction"


class TimeMode(str, Enum):
    INTELLIGENT = "INTELLIGENT"
    HARD = "HARD"


class TimestampedWindow(BaseModel):
    from_: datetime
    to: datetime

    class Config:
        fields = {"from_": "from"}
