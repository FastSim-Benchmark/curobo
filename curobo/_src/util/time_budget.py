# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Cooperative, call-local wall-clock budgets shared by nested planning stages."""

import math
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from numbers import Real
from time import monotonic
from typing import Callable

_deadline: ContextVar[float | None] = ContextVar("curobo_planning_deadline", default=None)


def check_time_budget(
    finish_pending: Callable[[], None] | None = None, *, context: str | None = None
) -> None:
    """Check the active deadline after optionally finishing one queued work chunk."""
    deadline = _deadline.get()
    if deadline is None:
        return
    if finish_pending is not None:
        finish_pending()
    if monotonic() >= deadline:
        detail = "" if context is None else f" ({context})"
        raise TimeoutError("cuRobo planning timeout_s budget exceeded" + detail)


@contextmanager
def time_budget(timeout_s: float | None):
    """Install a deadline that nested stages can shorten but cannot extend."""
    parent = _deadline.get()
    deadline = parent
    if timeout_s is not None:
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, Real):
            raise ValueError("timeout_s must be a finite positive number or None")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be a finite positive number or None")
        requested = monotonic() + timeout_s
        deadline = requested if parent is None else min(parent, requested)
    token = _deadline.set(deadline)
    try:
        check_time_budget()
        yield
        check_time_budget()
    finally:
        _deadline.reset(token)


def planning_time_budget(method):
    """Apply a public planner's optional keyword-only timeout to all child stages."""

    @wraps(method)
    def wrapped(*args, **kwargs):
        with time_budget(kwargs.get("timeout_s")):
            return method(*args, **kwargs)

    return wrapped
