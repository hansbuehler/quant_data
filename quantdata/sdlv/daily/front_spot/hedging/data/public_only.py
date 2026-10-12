"""Runtime guard for the distributable hedging dataset boundary."""

from __future__ import annotations

from quantdata.hedging.data.public_dataset import HEDGING_PUBLIC_SCHEMA
from quantdata.sdlv.daily.front_spot.hedging.data.public_view import (
    PublicHedgingPanelView,
)


def require_public_hedging_panel(panel: PublicHedgingPanelView) -> None:
    """Reject model or training inputs outside the distributable panel.

    The private-source exporter is deliberately separate from the runtime
    benchmark. Once the public candidate exists, every model fit, generated
    scenario, market mark, and off-grid interpolation must be reproducible from
    this panel alone.
    """
    if not isinstance(panel, PublicHedgingPanelView):
        raise TypeError(
            "hedging models and training require PublicHedgingPanelView; "
            "private-source loaders are not accepted"
        )
    schema = panel.geometry.get("schema")
    if schema != HEDGING_PUBLIC_SCHEMA:
        raise ValueError(
            f"unexpected hedging dataset schema {schema!r}; "
            f"expected {HEDGING_PUBLIC_SCHEMA!r}"
        )
