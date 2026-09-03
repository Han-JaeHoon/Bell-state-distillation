"""Shared random-state generators for tests (deterministic seeds only)."""

import numpy as np


def random_bell_diagonal_populations(rng, n):
    """Dirichlet-distributed Bell populations, shape (n, 4)."""
    return rng.dirichlet(np.ones(4), size=n)


def random_density_matrix(rng, dim=4):
    """Haar-ish random mixed state via a Ginibre ensemble."""
    a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    m = a @ a.conj().T
    return m / np.trace(m)
