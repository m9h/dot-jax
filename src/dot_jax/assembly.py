"""FEM stiffness matrix assembly.

JAX-native assembly of the CW diffusion system matrix A = K + M + C.
All functions are JIT-compatible and differentiable via jax.grad.

Functions:
    assemble_stiffness: Diffusion stiffness matrix K
    assemble_mass: Consistent absorption mass matrix M
    assemble_boundary: Robin boundary condition matrix C
    assemble_system_cw: Full CW system matrix A = K + M + C
"""

import jax.numpy as jnp

from .analytical import getreff


def assemble_stiffness(mesh, D):
    """Assemble diffusion stiffness matrix K.

    K_ij = sum_e D_e * (grad(phi_i) . grad(phi_j)) * V_e

    Parameters
    ----------
    mesh : FEMMesh
    D : float or (ne,) — diffusion coefficient per element.
        D = 1/(3*(mua + musp)).

    Returns
    -------
    K : (nn, nn) — symmetric positive semi-definite stiffness matrix.
    """
    nn = mesh.nn
    elem = mesh.elem
    dd = mesh.deldotdel

    D = jnp.atleast_1d(jnp.asarray(D, dtype=jnp.float64))
    if D.ndim == 0 or D.shape[0] == 1:
        D = jnp.broadcast_to(D.ravel()[0], (mesh.ne,))

    K = jnp.zeros((nn, nn))

    k = 0
    for i in range(4):
        for j in range(i, 4):
            rows = elem[:, i]
            cols = elem[:, j]
            vals = D * dd[:, k]
            K = K.at[rows, cols].add(vals)
            if i != j:
                K = K.at[cols, rows].add(vals)
            k += 1

    return K


def assemble_mass(mesh, mua):
    """Assemble consistent absorption mass matrix M.

    Uses linear tet consistent mass: coeff = 1/10 (diag), 1/20 (off-diag).
    M_ij = sum_e mua_e * V_e * coeff(i,j)

    Parameters
    ----------
    mesh : FEMMesh
    mua : float or (ne,) — absorption coefficient per element.

    Returns
    -------
    M : (nn, nn) — symmetric positive semi-definite mass matrix.
    """
    nn = mesh.nn
    elem = mesh.elem
    evol = mesh.evol

    mua = jnp.atleast_1d(jnp.asarray(mua, dtype=jnp.float64))
    if mua.ndim == 0 or mua.shape[0] == 1:
        mua = jnp.broadcast_to(mua.ravel()[0], (mesh.ne,))

    mua_evol = mua * evol

    M = jnp.zeros((nn, nn))

    for i in range(4):
        for j in range(i, 4):
            rows = elem[:, i]
            cols = elem[:, j]
            coeff = 0.10 if i == j else 0.05
            vals = mua_evol * coeff
            M = M.at[rows, cols].add(vals)
            if i != j:
                M = M.at[cols, rows].add(vals)

    return M


def assemble_boundary(mesh, n_in, n_out=1.0):
    """Assemble Robin boundary condition matrix C.

    Robin BC: phi + 2*A*D * dphi/dn = 0 at boundary.
    A = (1+Reff)/(1-Reff), giving surface integral coefficient
    bc_coeff = (1-Reff) / (12*(1+Reff)).

    Parameters
    ----------
    mesh : FEMMesh
    n_in : float — refractive index of tissue.
    n_out : float — refractive index of external medium.

    Returns
    -------
    C : (nn, nn) — symmetric positive semi-definite boundary matrix.
    """
    nn = mesh.nn
    face = mesh.face
    area = mesh.area

    Reff = getreff(n_in, n_out)
    bc_coeff = (1.0 - Reff) / (12.0 * (1.0 + Reff))

    C = jnp.zeros((nn, nn))

    for i in range(3):
        for j in range(i, 3):
            rows = face[:, i]
            cols = face[:, j]
            coeff = bc_coeff if i == j else bc_coeff * 0.5
            vals = area * coeff
            C = C.at[rows, cols].add(vals)
            if i != j:
                C = C.at[cols, rows].add(vals)

    return C


def assemble_system_cw(mesh, mua, musp, n_in=1.37, n_out=1.0):
    """Assemble full CW system matrix A = K + M + C.

    Parameters
    ----------
    mesh : FEMMesh
    mua : float or (ne,) — absorption coefficient (1/mm).
    musp : float or (ne,) — reduced scattering coefficient (1/mm).
    n_in : float — tissue refractive index.
    n_out : float — external refractive index.

    Returns
    -------
    A : (nn, nn) — symmetric positive definite system matrix.
    """
    mua = jnp.asarray(mua, dtype=jnp.float64)
    musp = jnp.asarray(musp, dtype=jnp.float64)
    D = 1.0 / (3.0 * (mua + musp))

    K = assemble_stiffness(mesh, D)
    M = assemble_mass(mesh, mua)
    C = assemble_boundary(mesh, n_in, n_out)

    return K + M + C
