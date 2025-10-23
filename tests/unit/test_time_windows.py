from __future__ import annotations

import datetime as dt

from app.utils.time_windows import (
    hard_mode_pre_mask,
    intelligent_decay,
    make_window,
    parse_date,
    window_iou,
)


def test_window_iou_overlap():
    left = make_window(parse_date("2025-01-01"), parse_date("2025-01-10"))
    right = make_window(parse_date("2025-01-05"), parse_date("2025-01-15"))
    assert 0 < window_iou(left, right) < 1


def test_hard_pre_mask_drops_non_overlap():
    left = make_window(parse_date("2025-01-01"), parse_date("2025-01-02"))
    right = make_window(parse_date("2025-02-01"), parse_date("2025-02-02"))
    assert not hard_mode_pre_mask(left, right)


def test_intelligent_decay_decreases_with_gap():
    left = make_window(parse_date("2025-01-01"), parse_date("2025-01-02"))
    right = make_window(parse_date("2025-01-03"), parse_date("2025-01-04"))
    near = intelligent_decay(left, right)
    far = intelligent_decay(left, make_window(parse_date("2025-06-01"), parse_date("2025-06-02")))
    assert near > far
