import jax
import jax.numpy as jnp
import numpy as np

from quantdata.hedging.model_audit.repository_gromm2_recursive import leading_normals


def test_recursive_uses_leading_pairs_of_full_static_random_stream():
    key = jax.random.PRNGKey(20_261_112)
    raw = jax.random.normal(key, (500, 17))
    full = jnp.stack([raw, -raw], axis=1).reshape(1000, 17)
    selected = leading_normals(key, total_paths=1000, paths=32, factors=16)
    np.testing.assert_array_equal(np.asarray(selected), np.asarray(full[:32]))
