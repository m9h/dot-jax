"""FEM stiffness matrix assembly.

JAX-native assembly of the CW diffusion system matrix A = K + M + C
for the photon diffusion equation:

    -div(D grad(phi)) + mua * phi = S

Discretised with linear tetrahedral elements. The stiffness matrix K
encodes diffusion, M encodes absorption via consistent mass, and C
enforces the Robin (type III) boundary condition. Cross-validated
against redbirdpy's ``femlhs``.

The Robin BC follows Haskell et al. (1994) using the effective
reflection coefficient for the extrapolated boundary.

References
----------
.. [1] S. R. Arridge, M. Schweiger, M. Hiraoka, and D. T. Delpy,
       "A finite element approach for modeling photon transport in
       tissue," Med. Phys., vol. 20, no. 2, pp. 299-309, 1993.
.. [2] M. Schweiger, S. R. Arridge, M. Hiraoka, and D. T. Delpy,
       "The finite element method for the propagation of light in
       scattering media," Med. Phys., vol. 20, pp. 1203-1210, 1993.

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

    # Loop over the 10 unique pairs (i, j) with i <= j among the 4
    # tet basis functions.  dd[:, k] stores the pre-integrated product
    # grad(phi_i) . grad(phi_j) * V_e for each element.  Multiplying
    # by D_e gives the element contribution to the global stiffness
    # matrix.  The scatter-add (.at[].add) assembles these into the
    # global (nn x nn) matrix.  Off-diagonal entries are symmetric,
    # so we add to both (i,j) and (j,i).
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

    # Product mua_e * V_e weights the consistent mass contribution of
    # each element.
    mua_evol = mua * evol

    M = jnp.zeros((nn, nn))

    # Linear tetrahedral consistent mass coefficients:
    #   diagonal (i == j): integral of phi_i^2 over tet = V_e / 10
    #   off-diag (i != j): integral of phi_i * phi_j   = V_e / 20
    # The 0.10 and 0.05 below already incorporate the 1/V_e factor
    # that cancels against the V_e in mua_evol (net: mua * V_e * coeff).
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

    # Effective internal reflection coefficient from Haskell et al. (1994).
    Reff = getreff(n_in, n_out)

    # Robin BC surface integral coefficient.
    # From the extrapolated boundary condition:
    #   phi + 2*A*D * dphi/dn = 0,  A = (1+Reff)/(1-Reff)
    # The surface mass matrix entry for triangular faces uses:
    #   bc_coeff = (1 - Reff) / (12 * (1 + Reff))
    bc_coeff = (1.0 - Reff) / (12.0 * (1.0 + Reff))

    C = jnp.zeros((nn, nn))

    # Assemble over boundary triangular faces.  The consistent mass
    # coefficients for a linear triangle are:
    #   diagonal (i == j):  area_f * bc_coeff
    #   off-diag (i != j):  area_f * bc_coeff * 0.5
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

    # Diffusion coefficient: D = 1 / (3 * (mua + musp))
    D = 1.0 / (3.0 * (mua + musp))

    # Stiffness K encodes diffusion (gradient-gradient bilinear form),
    # Mass M encodes absorption, and Boundary C enforces the Robin BC
    # on the extrapolated boundary.
    K = assemble_stiffness(mesh, D)
    M = assemble_mass(mesh, mua)
    C = assemble_boundary(mesh, n_in, n_out)

    # The combined system A = K + M + C is symmetric positive definite
    # when mua > 0 and musp > 0, ensuring a unique solution to the
    # CW diffusion equation.
    return K + M + C
