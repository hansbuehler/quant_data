import numpy as np
import pytest

from quantdata.dysanos.grid.gendysanos.run_surface_first_diagnostics import (
    _align_eigensurface_signs,
    _leading_pca,
    _rankwise_component_angles_degrees,
)


def test_leading_pca_returns_ordered_orthonormal_components():
    rng = np.random.default_rng(812)
    scales = np.array([4.0, 2.0, 1.0, 0.5])
    changes = rng.normal(size=(20_000, scales.size)) * scales

    result = _leading_pca(changes, components=3)
    components = result["components"]
    ratios = result["explained_variance_ratio"]

    assert components @ components.T == pytest.approx(np.eye(3), abs=1.0e-12)
    assert np.all(np.diff(ratios) < 0.0)
    assert abs(components[0, 0]) > 0.999
    assert ratios == pytest.approx(
        np.square(scales[:3]) / np.sum(np.square(scales)),
        abs=0.006,
    )


def test_eigensurface_sign_alignment_preserves_directions():
    reference = np.array(
        [
            [-0.8, 0.6, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )
    candidate = -reference

    aligned_reference, aligned_candidate = _align_eigensurface_signs(
        reference,
        candidate,
    )

    assert aligned_reference[0] == pytest.approx(-reference[0])
    assert aligned_reference[1] == pytest.approx(-reference[1])
    assert aligned_candidate == pytest.approx(aligned_reference)
    assert _rankwise_component_angles_degrees(
        aligned_reference,
        aligned_candidate,
    ) == pytest.approx(np.zeros(2))
