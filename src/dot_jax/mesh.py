"""FEM mesh data structures and gradient assembly.

JAX-native tetrahedral mesh module. FEMMesh is an Equinox Module holding
precomputed mesh operators. All indices are 0-based (dot-jax convention).

Functions:
    compute_evol: Element volumes for tetrahedra
    compute_face_area: Triangle face areas
    compute_deldotdel: grad(phi_i)·grad(phi_j) per element
    compute_nvol: Nodal volumes via scatter-add
    elem2node: Scatter-add element values to nodes
    extract_surface: Extract boundary faces (numpy, outside JIT)
    reorient_elems: Ensure positive element volumes (numpy, outside JIT)
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


class FEMMesh(eqx.Module):
    """Tetrahedral FEM mesh with precomputed operators.

    All indices are 0-based. Constructed via ``FEMMesh.create()``.

    Attributes
    ----------
    node : (nn, 3) float64 — node coordinates
    elem : (ne, 4) int32 — element connectivity (0-based)
    face : (nf, 3) int32 — boundary face connectivity (0-based)
    evol : (ne,) float64 — element volumes
    area : (nf,) float64 — face areas
    nvol : (nn,) float64 — nodal volumes
    deldotdel : (ne, 10) float64 — grad(phi_i)·grad(phi_j) products
    """
    node: jnp.ndarray
    elem: jnp.ndarray
    face: jnp.ndarray
    evol: jnp.ndarray
    area: jnp.ndarray
    nvol: jnp.ndarray
    deldotdel: jnp.ndarray

    @staticmethod
    def create(node, elem, face=None):
        """Build a FEMMesh from node/elem arrays (0-based).

        Parameters
        ----------
        node : array, shape (nn, 3)
            Node coordinates.
        elem : array, shape (ne, 4)
            Element connectivity (0-based).
        face : array, shape (nf, 3), optional
            Boundary faces (0-based). Extracted automatically if None.
        """
        node = jnp.asarray(node, dtype=jnp.float64)
        elem = jnp.asarray(elem, dtype=jnp.int32)

        # Reorient elements to have positive volume
        elem_np = np.asarray(elem)
        node_np = np.asarray(node)
        elem_np = reorient_elems(node_np, elem_np)
        elem = jnp.asarray(elem_np, dtype=jnp.int32)

        if face is None:
            face_np = extract_surface(elem_np)
            face = jnp.asarray(face_np, dtype=jnp.int32)
        else:
            face = jnp.asarray(face, dtype=jnp.int32)

        evol = compute_evol(node, elem)
        area = compute_face_area(node, face)
        nvol = compute_nvol(elem, evol, node.shape[0])
        dd, _ = compute_deldotdel(node, elem, evol)

        return FEMMesh(
            node=node, elem=elem, face=face,
            evol=evol, area=area, nvol=nvol, deldotdel=dd,
        )

    @property
    def nn(self):
        return self.node.shape[0]

    @property
    def ne(self):
        return self.elem.shape[0]

    @property
    def nf(self):
        return self.face.shape[0]


# =============================================================================
# JIT-compatible mesh computations
# =============================================================================


def compute_evol(node, elem):
    """Compute tetrahedral element volumes.

    Parameters
    ----------
    node : (nn, 3) node coordinates.
    elem : (ne, 4) element connectivity (0-based).

    Returns
    -------
    evol : (ne,) element volumes.
    """
    n0 = node[elem[:, 0]]
    v1 = node[elem[:, 1]] - n0
    v2 = node[elem[:, 2]] - n0
    v3 = node[elem[:, 3]] - n0
    return jnp.abs(jnp.sum(jnp.cross(v1, v2) * v3, axis=1)) / 6.0


def compute_face_area(node, face):
    """Compute triangle face areas.

    Parameters
    ----------
    node : (nn, 3) node coordinates.
    face : (nf, 3) face connectivity (0-based).

    Returns
    -------
    area : (nf,) face areas.
    """
    v1 = node[face[:, 1]] - node[face[:, 0]]
    v2 = node[face[:, 2]] - node[face[:, 0]]
    return 0.5 * jnp.sqrt(jnp.sum(jnp.cross(v1, v2) ** 2, axis=1))


def compute_deldotdel(node, elem, evol):
    """Compute grad(phi_i)·grad(phi_j) for each tetrahedral element.

    Ports the redbirdpy deldotdel algorithm exactly. For each element,
    computes the 10 unique dot products of the 4 basis function gradients.

    Parameters
    ----------
    node : (nn, 3) node coordinates.
    elem : (ne, 4) element connectivity (0-based).
    evol : (ne,) element volumes.

    Returns
    -------
    dd : (ne, 10) upper-triangle dot products,
        ordered (0,0),(0,1),(0,2),(0,3),(1,1),(1,2),(1,3),(2,2),(2,3),(3,3).
    delphi : (ne, 3, 4) gradient components per element.
    """
    # Node coordinates per element: (ne, 4, 3) → transpose to (ne, 3, 4)
    no = node[elem].transpose(0, 2, 1)

    evol_inv = 1.0 / (evol * 6.0)

    # Cross-product column indices (matches redbirdpy exactly)
    col = jnp.array([[3, 1, 2, 1], [2, 0, 3, 2], [1, 3, 0, 3], [0, 2, 1, 0]])

    delphi = jnp.zeros((node.shape[0], 3, 4)).at[:1].get()  # dummy for shape
    # Build delphi: (ne, 3, 4)
    parts = []
    for coord in range(3):
        idx = [c for c in range(3) if c != coord]
        coord_parts = []
        for i in range(4):
            term1 = no[:, idx[0], col[i, 0]] - no[:, idx[0], col[i, 1]]
            term2 = no[:, idx[1], col[i, 2]] - no[:, idx[1], col[i, 3]]
            term3 = no[:, idx[0], col[i, 2]] - no[:, idx[0], col[i, 3]]
            term4 = no[:, idx[1], col[i, 0]] - no[:, idx[1], col[i, 1]]
            coord_parts.append((term1 * term2 - term3 * term4) * evol_inv)
        parts.append(jnp.stack(coord_parts, axis=1))  # (ne, 4)

    delphi = jnp.stack(parts, axis=1)  # (ne, 3, 4)

    # 10 unique dot products: upper triangle of (i, j) for 4 basis functions
    pairs = []
    for i in range(4):
        for j in range(i, 4):
            pairs.append(jnp.sum(delphi[:, :, i] * delphi[:, :, j], axis=1))

    dd = jnp.stack(pairs, axis=1)  # (ne, 10)
    dd = dd * evol[:, None]

    return dd, delphi


def compute_nvol(elem, evol, nn):
    """Compute nodal volumes (1/4 of connected element volumes).

    Parameters
    ----------
    elem : (ne, 4) element connectivity (0-based).
    evol : (ne,) element volumes.
    nn : int — number of nodes.

    Returns
    -------
    nvol : (nn,) nodal volumes.
    """
    nvol = jnp.zeros(nn, dtype=evol.dtype)
    for j in range(4):
        nvol = nvol.at[elem[:, j]].add(evol)
    return nvol * 0.25


def elem2node(elem, elemval, nn):
    """Scatter-add element values to nodes (average over 4 vertices).

    Parameters
    ----------
    elem : (ne, 4) element connectivity (0-based).
    elemval : (ne,) or (ne, nval) element-based values.
    nn : int — number of nodes.

    Returns
    -------
    nodeval : (nn,) or (nn, nval) node-averaged values.
    """
    squeeze = elemval.ndim == 1
    if squeeze:
        elemval = elemval[:, None]

    nval = elemval.shape[1]
    nodeval = jnp.zeros((nn, nval), dtype=elemval.dtype)

    for j in range(4):
        nodeval = nodeval.at[elem[:, j]].add(elemval)

    nodeval = nodeval * 0.25

    if squeeze:
        return nodeval[:, 0]
    return nodeval


# =============================================================================
# Mesh preprocessing (numpy, outside JIT)
# =============================================================================


def reorient_elems(node, elem):
    """Reorient elements to have positive volume. 0-based indices.

    Parameters
    ----------
    node : (nn, 3) ndarray
    elem : (ne, 4) ndarray, 0-based

    Returns
    -------
    elem : (ne, 4) ndarray with consistent orientation.
    """
    elem = np.array(elem, copy=True)
    n0 = node[elem[:, 0]]
    v1 = node[elem[:, 1]] - n0
    v2 = node[elem[:, 2]] - n0
    v3 = node[elem[:, 3]] - n0
    vol = np.sum(np.cross(v1, v2) * v3, axis=1)

    neg = vol < 0
    elem[neg, 0], elem[neg, 1] = elem[neg, 1].copy(), elem[neg, 0].copy()
    return elem


def extract_surface(elem):
    """Extract boundary triangular faces from tetrahedral mesh. 0-based.

    Parameters
    ----------
    elem : (ne, 4) ndarray, 0-based

    Returns
    -------
    face : (nf, 3) ndarray, 0-based boundary faces.
    """
    elem = np.asarray(elem)[:, :4]

    # 4 faces per tet, oriented outward
    faces = np.vstack([
        elem[:, [0, 2, 1]],
        elem[:, [0, 1, 3]],
        elem[:, [0, 3, 2]],
        elem[:, [1, 2, 3]],
    ])

    faces_sorted = np.sort(faces, axis=1)
    _, indices, counts = np.unique(
        faces_sorted, axis=0, return_index=True, return_counts=True,
    )

    boundary_idx = indices[counts == 1]
    return faces[boundary_idx]
