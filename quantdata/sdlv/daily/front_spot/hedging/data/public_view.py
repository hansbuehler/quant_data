"""Nominal marker for public-only hedging panel views."""

from __future__ import annotations


class PublicHedgingPanelView:
    """Base class for unrestricted or date-restricted public panel views."""

    dataset_manifest_sha256: str
