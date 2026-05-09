'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: Shared Pydantic schemas and enumerations for the production lane.

These types are used across bundle manifests, serving.duckdb, worker runs,
API responses, and portfolio analysis.  Keeping them in one place ensures the
whole production stack speaks the same vocabulary.

Reference: rpi5.md §10.4, §16, §9
'''

from __future__ import annotations

from enum import Enum


class PredictedClass(str, Enum):
    """The three canonical volatility-regime classes (rpi5.md §16)."""

    low = "low"
    medium = "medium"
    high = "high"


class RunStatus(str, Enum):
    """Per-asset daily run status codes (rpi5.md §10.4)."""

    fresh = "fresh"           # inference ran with fully up-to-date data
    stale_news = "stale_news" # inference ran but GKG/news data is stale
    stale_data = "stale_data" # inference ran but market/macro data is stale
    failed = "failed"         # inference did not produce a result


class PromotionStatus(str, Enum):
    """Outcome of a bundle promotion attempt (rpi5.md §9)."""

    promoted = "promoted"
    failed = "failed"
    skipped = "skipped"       # validation passed but promotion was intentionally deferred


class UserRole(str, Enum):
    """Application user roles (rpi5.md §14.3)."""

    user = "user"
    admin = "admin"
