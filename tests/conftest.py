import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def key():
    """Fixed PRNG key for reproducibility."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def tissue_props():
    """Standard tissue optical properties for testing."""
    return {
        "mua": 0.01,
        "musp": 1.0,
        "n_in": 1.37,
        "n_out": 1.0,
        "g": 0.0,
    }


@pytest.fixture
def simple_positions():
    """Standard source/detector positions for testing."""
    return {
        "srcpos": jnp.array([[0.0, 0.0, 0.0]]),
        "detpos": jnp.array([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
        ]),
    }


@pytest.fixture
def tiny_mesh():
    """Minimal 5-element tetrahedral mesh (0-based) for unit tests.

    A small box discretised into 5 tetrahedra. All indices are 0-based
    (dot-jax convention).
    """
    node = jnp.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [10.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
        [10.0, 0.0, 10.0],
        [0.0, 10.0, 10.0],
        [10.0, 10.0, 10.0],
    ], dtype=jnp.float64)

    elem = jnp.array([
        [0, 1, 2, 4],
        [1, 2, 3, 7],
        [1, 2, 4, 7],
        [1, 4, 5, 7],
        [2, 4, 6, 7],
    ], dtype=jnp.int32)

    face = jnp.array([
        [0, 1, 2],
        [1, 2, 3],
        [0, 1, 4],
        [1, 4, 5],
        [0, 2, 4],
        [2, 4, 6],
        [1, 3, 5],
        [3, 5, 7],
        [2, 3, 6],
        [3, 6, 7],
        [4, 5, 6],
        [5, 6, 7],
    ], dtype=jnp.int32)

    return {"node": node, "elem": elem, "face": face}
