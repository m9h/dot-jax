"""Differentiable FEM forward model for diffuse optical tomography.

JAX-native CW forward solver using Lineax for the linear system.
Source/detector placement uses barycentric interpolation. Detector
values are extracted via the adjoint method (rhs_det^T @ phi_src),
which is equivalent to the reciprocity principle in photon transport.

The full ``forward_cw`` pipeline is differentiable w.r.t. optical
properties (mua, musp), enabling autodiff Jacobians for reconstruction.

References
----------
.. [1] S. R. Arridge, "Optical tomography in medical imaging,"
       Inverse Problems, vol. 15, no. 2, pp. R41-R93, 1999.
.. [2] P. Kidger, "On neural differential equations," DPhil thesis,
       University of Oxford, 2021. (Lineax/Equinox framework)

Functions:
    locate_sources: Find containing elements and barycentric coords
    assemble_rhs: Build RHS vectors from point positions
    get_detector_values: Extract detector values via adjoint method
    forward_cw: Full CW forward solve pipeline
"""

import jax.numpy as jnp
import lineax as lx
import numpy as np

from ._types import ForwardResult
from .assembly import assemble_system_cw


def locate_sources(mesh, positions):
    """Find containing element and barycentric coordinates for points.

    Uses numpy (outside JIT) to locate each point in the mesh.

    Parameters
    ----------
    mesh : FEMMesh
    positions : (n_pts, 3) — point coordinates.

    Returns
    -------
    elem_idx : (n_pts,) int — containing element index (0-based).
    bary : (n_pts, 4) float — barycentric coordinates.
    """
    node = np.asarray(mesh.node)
    elem = np.asarray(mesh.elem)
    positions = np.atleast_2d(np.asarray(positions))
    n_pts = positions.shape[0]

    elem_idx = np.zeros(n_pts, dtype=np.int32)
    bary = np.zeros((n_pts, 4), dtype=np.float64)

    for p in range(n_pts):
        pt = positions[p]
        best_elem = 0
        best_bary = np.zeros(4)
        best_min = -np.inf

        for e in range(elem.shape[0]):
            n0, n1, n2, n3 = node[elem[e]]
            # Solve for barycentric coords: pt = l0*n0 + l1*n1 + l2*n2 + l3*n3
            # with l0 + l1 + l2 + l3 = 1
            T = np.column_stack([n1 - n0, n2 - n0, n3 - n0])
            try:
                lam = np.linalg.solve(T, pt - n0)
            except np.linalg.LinAlgError:
                continue
            l0 = 1.0 - lam[0] - lam[1] - lam[2]
            b = np.array([l0, lam[0], lam[1], lam[2]])

            # Pick element where all bary coords are most interior
            min_b = np.min(b)
            if min_b > best_min:
                best_min = min_b
                best_elem = e
                best_bary = b

        elem_idx[p] = best_elem
        bary[p] = np.maximum(best_bary, 0.0)
        bary[p] /= bary[p].sum()  # renormalize

    return jnp.array(elem_idx), jnp.array(bary)


def assemble_rhs(mesh, positions):
    """Build RHS source vectors from point positions.

    Each column is a source vector using barycentric interpolation:
    rhs[node_i, src_k] = bary_i for the containing element's nodes.

    Parameters
    ----------
    mesh : FEMMesh
    positions : (n_pts, 3) — source/detector positions.

    Returns
    -------
    rhs : (nn, n_pts) — RHS matrix.
    """
    elem_idx, bary = locate_sources(mesh, positions)
    elem = mesh.elem
    nn = mesh.nn
    n_pts = bary.shape[0]

    rhs = jnp.zeros((nn, n_pts))
    for p in range(n_pts):
        e = elem_idx[p]
        nodes = elem[e]
        for j in range(4):
            rhs = rhs.at[nodes[j], p].set(bary[p, j])

    return rhs


def get_detector_values(phi, rhs_det):
    """Extract detector measurements via adjoint method.

    Parameters
    ----------
    phi : (nn, n_src) — fluence field from forward solve.
    rhs_det : (nn, n_det) — detector RHS vectors.

    Returns
    -------
    detval : (n_det, n_src) — detector measurements.
    """
    return rhs_det.T @ phi


def forward_cw(mesh, mua, musp, srcpos, detpos, n_in=1.37, n_out=1.0):
    """CW forward solve: assemble, solve, extract.

    Parameters
    ----------
    mesh : FEMMesh
    mua : float — absorption coefficient (1/mm).
    musp : float — reduced scattering coefficient (1/mm).
    srcpos : (n_src, 3) — source positions.
    detpos : (n_det, 3) — detector positions.
    n_in : float — tissue refractive index.
    n_out : float — external refractive index.

    Returns
    -------
    ForwardResult(detval, phi)
        detval : (n_det, n_src) — detector measurements.
        phi : (nn, n_src) — fluence field at all nodes.
    """
    # Assemble system matrix (differentiable w.r.t. mua, musp)
    A = assemble_system_cw(mesh, mua, musp, n_in, n_out)

    # Assemble source/detector RHS (not differentiable — positions are fixed)
    rhs_src = assemble_rhs(mesh, srcpos)
    rhs_det = assemble_rhs(mesh, detpos)

    # Solve via Lineax (one column per source)
    import jax
    operator = lx.MatrixLinearOperator(A)

    def solve_col(b):
        return lx.linear_solve(operator, b, solver=lx.LU()).value

    phi = jax.vmap(solve_col, in_axes=1, out_axes=1)(rhs_src)

    # Extract detector values
    detval = get_detector_values(phi, rhs_det)

    return ForwardResult(detval=detval, phi=phi)
