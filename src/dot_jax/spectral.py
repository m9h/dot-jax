"""Multi-spectral chromophore decomposition (Beer-Lambert law).

JAX-native spectral forward model running CW forward solves across
wavelengths. Supports autodiff Jacobians for reconstruction.

The absorption Jacobian uses the adjoint (Frechet derivative) formula:
    J[d,s,n] = -phi_src[n,s] * phi_det[n,d] * V_n
where phi_src and phi_det are the forward and adjoint Green's functions.
This is the standard sensitivity kernel for DOT (Arridge, 1999).

Multi-wavelength operation enables spectral decomposition of
chromophore concentrations via the modified Beer-Lambert law,
following the approach of Corlu et al. (2003) and Dehghani et al. (2009).

References
----------
.. [1] A. Corlu et al., "Three-dimensional in vivo fluorescence
       diffuse optical tomography of breast cancer in humans,"
       Opt. Express, vol. 15, pp. 6696-6716, 2007.
.. [2] H. Dehghani et al., "Near infrared optical tomography using
       NIRFAST," Int. J. Numer. Methods Biomed. Eng., vol. 25,
       pp. 711-732, 2009.

Functions:
    spectral_forward_cw: Forward solve at multiple wavelengths
    compute_jacobian_mua: Sensitivity matrix d(detval)/d(mua_node)
    normalize_jacobian: Column-norm depth weighting for Jacobian
"""

import jax
import jax.numpy as jnp

from .forward import forward_cw, assemble_rhs, get_detector_values
from .assembly import assemble_system_cw


def spectral_forward_cw(mesh, mua_wv, musp_wv, srcpos, detpos,
                         n_in=1.37, n_out=1.0):
    """CW forward solve at multiple wavelengths.

    Parameters
    ----------
    mesh : FEMMesh
    mua_wv : (n_wv,) — absorption coefficient at each wavelength.
    musp_wv : (n_wv,) — reduced scattering at each wavelength.
    srcpos : (n_src, 3) — source positions.
    detpos : (n_det, 3) — detector positions.
    n_in, n_out : float — refractive indices.

    Returns
    -------
    detvals : (n_wv, n_det, n_src) — detector measurements per wavelength.
    """
    n_wv = mua_wv.shape[0]

    results = []
    for w in range(n_wv):
        r = forward_cw(mesh, mua_wv[w], musp_wv[w], srcpos, detpos, n_in, n_out)
        results.append(r.detval)

    return jnp.stack(results, axis=0)


def compute_jacobian_mua(mesh, mua, musp, srcpos, detpos,
                          n_in=1.37, n_out=1.0):
    """Compute Jacobian of detector values w.r.t. nodal absorption.

    Uses the adjoint method: J = -phi_src * phi_det (Rytov/Born approx).

    Parameters
    ----------
    mesh : FEMMesh
    mua : float — baseline absorption coefficient.
    musp : float — baseline reduced scattering coefficient.
    srcpos : (n_src, 3) — source positions.
    detpos : (n_det, 3) — detector positions.
    n_in, n_out : float — refractive indices.

    Returns
    -------
    J : (n_meas, nn) — Jacobian matrix.
        n_meas = n_det * n_src. Rows ordered detector-major.
    """
    import lineax as lx

    # Assemble system
    A = assemble_system_cw(mesh, mua, musp, n_in, n_out)
    operator = lx.MatrixLinearOperator(A)

    def solve_col(b):
        return lx.linear_solve(operator, b, solver=lx.LU()).value

    # Solve forward (sources) and adjoint (detectors)
    rhs_src = assemble_rhs(mesh, srcpos)
    rhs_det = assemble_rhs(mesh, detpos)

    phi_src = jax.vmap(solve_col, in_axes=1, out_axes=1)(rhs_src)   # (nn, n_src)
    phi_det = jax.vmap(solve_col, in_axes=1, out_axes=1)(rhs_det)   # (nn, n_det)

    n_src = phi_src.shape[1]
    n_det = phi_det.shape[1]

    # Adjoint (Frechet derivative) formula for the absorption Jacobian:
    #   J[d*n_src + s, n] = -phi_det[n, d] * phi_src[n, s] * V_n
    # where phi_src is the forward Green's function (source → nodes),
    # phi_det is the adjoint Green's function (detector → nodes), and
    # V_n is the nodal volume.  Rows are ordered detector-major:
    # (det0,src0), (det0,src1), ..., (det1,src0), ...
    rows = []
    for d in range(n_det):
        for s in range(n_src):
            row = -phi_det[:, d] * phi_src[:, s] * mesh.nvol
            rows.append(row)

    return jnp.stack(rows, axis=0)


def normalize_jacobian(J):
    """Column-normalize Jacobian for depth-weighted reconstruction.

    Divides each column by its L2 norm to compensate the surface bias
    inherent in DOT sensitivity. Deeper nodes have smaller Jacobian
    columns; normalization equalizes their contribution to the inversion.

    After reconstruction with J_norm, scale back via:
        delta_mua_true = delta_mua_norm / col_norms

    Parameters
    ----------
    J : (n_meas, n_nodes) — raw Jacobian matrix.

    Returns
    -------
    J_norm : (n_meas, n_nodes) — column-normalized Jacobian.
    col_norms : (n_nodes,) — original column L2 norms.
    """
    col_norms = jnp.linalg.norm(J, axis=0)
    safe_norms = jnp.where(col_norms > 1e-30, col_norms, 1.0)
    J_norm = J / safe_norms[None, :]
    return J_norm, col_norms
