"""Geometry selection shared by DLV initialization, fitting and audits."""

from dataclasses import replace


def with_model_business_dtes(geometry, dtes: list[int] | None, *, allow_zero_backbone: bool = False):
    """Return the requested expiry geometry, retaining the source by default."""
    if allow_zero_backbone:
        # The geometry enforces a strictly positive floor; this approximates
        # the zero-backbone boundary without changing its parameter invariant.
        geometry = replace(geometry, min_eta=1e-8)
    if dtes is None:
        return geometry
    if not dtes or any(day <= 0 for day in dtes) or any(a >= b for a, b in zip(dtes, dtes[1:])):
        raise ValueError("Model expiry days must be positive and strictly increasing")
    return replace(geometry, num_model_expiries=len(dtes), explicit_model_business_dtes=tuple(map(float, dtes)))
