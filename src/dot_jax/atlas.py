"""Atlas-based head mesh generation for DOT forward modelling.

Generates tetrahedral head meshes from the MNI152 atlas using
scikit-image marching cubes and scipy Delaunay tetrahedralization.
Pure Python — no external meshing binaries required (works on ARM64).

The pipeline:
1. Fetch MNI152 tissue probability maps (CSF, GM, WM) via nilearn
2. Create a brain mask from the combined tissue maps
3. Extract surface via marching cubes
4. Fill interior with Delaunay tetrahedralization
5. Convert to FEMMesh with MNI/Talairach coordinates

Functions:
    fetch_mni152_seg: Fetch MNI152 tissue probability maps
    generate_head_mesh: Generate tetrahedral head mesh from atlas
    project_to_surface: Project optode positions to mesh surface
"""

import numpy as np
from scipy.spatial import Delaunay, cKDTree
from skimage.measure import marching_cubes

from .mesh import FEMMesh


def fetch_mni152_seg():
    """Fetch MNI152 tissue probability maps (CSF, GM, WM).

    Returns
    -------
    seg : (197, 233, 189, 3) ndarray
        Tissue probability maps: [:,:,:,0]=CSF, [:,:,:,1]=GM, [:,:,:,2]=WM.
        Values in [0, 1]. 1mm isotropic in MNI space.
    """
    import nibabel as nib
    from nilearn.datasets import fetch_icbm152_2009

    mni = fetch_icbm152_2009()
    csf = nib.load(mni["csf"]).get_fdata()
    gm = nib.load(mni["gm"]).get_fdata()
    wm = nib.load(mni["wm"]).get_fdata()

    return np.stack([csf, gm, wm], axis=-1)


def generate_head_mesh(seg=None, max_nodes=5000, max_elem_vol=100,
                       threshold=0.3, decimate_target=None):
    """Generate a tetrahedral head mesh from MNI152 tissue maps.

    Uses marching cubes for surface extraction and scipy Delaunay for
    tetrahedralization. Coordinates are in MNI/Talairach space (mm).

    Parameters
    ----------
    seg : (X, Y, Z, 3) ndarray, optional
        Tissue probability maps. Fetched from nilearn if None.
    max_nodes : int
        Target maximum node count (controls surface + interior density).
    max_elem_vol : float
        Maximum element volume (mm^3) for interior point sampling.
    threshold : float
        Probability threshold for brain mask (sum of tissue probs).
    decimate_target : int, optional
        Target surface vertex count after decimation. Defaults to max_nodes // 2.

    Returns
    -------
    FEMMesh
        Tetrahedral mesh in MNI coordinates (mm), 0-based indexing.
    """
    if seg is None:
        seg = fetch_mni152_seg()

    if decimate_target is None:
        decimate_target = max_nodes // 2

    # MNI152 affine: origin at (-98, -134, -72), 1mm isotropic
    origin = np.array([-98.0, -134.0, -72.0])

    # Combined brain mask
    brain_prob = seg.sum(axis=-1)
    brain_mask = brain_prob > threshold

    # Extract surface with marching cubes (in voxel coords)
    verts_vox, faces, _, _ = marching_cubes(brain_prob, level=threshold)

    # Convert to MNI coordinates (mm)
    verts_mni = verts_vox + origin

    # Subsample surface if too many vertices
    if len(verts_mni) > decimate_target:
        idx = _subsample_surface(verts_mni, decimate_target)
        verts_mni = verts_mni[idx]

    # Sample interior points
    interior_pts = _sample_interior(brain_mask, origin, max_elem_vol,
                                    max_nodes - len(verts_mni))

    # Combine surface + interior
    all_pts = np.vstack([verts_mni, interior_pts]) if len(interior_pts) > 0 else verts_mni

    # Delaunay tetrahedralization
    tri = Delaunay(all_pts)

    # Filter elements: keep only those with centroid inside the brain
    centroids_vox = tri.points[tri.simplices].mean(axis=1) - origin
    centroids_vox = np.clip(centroids_vox.astype(int), 0,
                            np.array(brain_mask.shape) - 1)
    inside = brain_mask[centroids_vox[:, 0], centroids_vox[:, 1], centroids_vox[:, 2]]
    elem = tri.simplices[inside]

    # Remove degenerate (zero-volume) elements
    n0 = all_pts[elem[:, 0]]
    v1 = all_pts[elem[:, 1]] - n0
    v2 = all_pts[elem[:, 2]] - n0
    v3 = all_pts[elem[:, 3]] - n0
    vol = np.abs(np.sum(np.cross(v1, v2) * v3, axis=1)) / 6.0
    elem = elem[vol > 1e-10]

    # Remove unused nodes
    used_nodes = np.unique(elem)
    node_map = np.full(len(all_pts), -1, dtype=np.int32)
    node_map[used_nodes] = np.arange(len(used_nodes))
    node = all_pts[used_nodes]
    elem = node_map[elem]

    return FEMMesh.create(node, elem.astype(np.int32))


def project_to_surface(mesh, positions):
    """Project points to the nearest surface node of a FEMMesh.

    Parameters
    ----------
    mesh : FEMMesh
    positions : (n_pts, 3) ndarray
        Points to project (e.g., optode positions in MNI space).

    Returns
    -------
    projected : (n_pts, 3) ndarray
        Positions projected to the mesh surface.
    """
    node = np.array(mesh.node)
    face = np.array(mesh.face)

    # Get unique surface nodes
    surface_node_idx = np.unique(face.ravel())
    surface_nodes = node[surface_node_idx]

    # KD-tree nearest-neighbour lookup
    tree = cKDTree(surface_nodes)
    _, idx = tree.query(np.asarray(positions))

    return surface_nodes[idx]


# =============================================================================
# Internal helpers
# =============================================================================


def _subsample_surface(verts, target):
    """Subsample surface vertices using farthest-point sampling.

    Greedily selects the vertex that maximises the minimum distance to
    all previously selected vertices, producing a well-spaced subset.

    Parameters
    ----------
    verts : (N, 3) ndarray
        Surface vertex positions.
    target : int
        Desired number of output vertices.

    Returns
    -------
    indices : (target,) ndarray of int
        Indices into *verts* of the selected subset.
    """
    n = len(verts)
    if n <= target:
        return np.arange(n)

    rng = np.random.default_rng(42)
    selected = [rng.integers(n)]
    dists = np.full(n, np.inf)

    for _ in range(target - 1):
        d = np.linalg.norm(verts - verts[selected[-1]], axis=1)
        dists = np.minimum(dists, d)
        selected.append(np.argmax(dists))

    return np.array(selected)


def _sample_interior(mask, origin, max_vol, max_pts):
    """Sample interior points on a regular grid within a binary mask.

    Generates candidate points on a grid whose spacing is derived from
    ``max_vol``, retains only those inside the mask, then randomly
    subsamples to at most ``max_pts``.

    Parameters
    ----------
    mask : (X, Y, Z) ndarray of bool
        Binary brain mask in voxel space.
    origin : (3,) ndarray
        MNI coordinate of voxel (0, 0, 0).
    max_vol : float
        Target maximum element volume (mm^3).  Controls grid spacing
        via ``spacing = max_vol^(1/3)``.
    max_pts : int
        Maximum number of interior points to return.

    Returns
    -------
    pts : (M, 3) ndarray
        Interior point coordinates in MNI space (mm).
    """
    if max_pts <= 0:
        return np.empty((0, 3))

    # Spacing from target volume: vol = spacing^3
    spacing = max(max_vol ** (1.0 / 3.0), 3.0)

    # Grid points inside mask
    shape = np.array(mask.shape)
    grid_x = np.arange(0, shape[0], spacing)
    grid_y = np.arange(0, shape[1], spacing)
    grid_z = np.arange(0, shape[2], spacing)

    xx, yy, zz = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
    grid_pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    # Filter to inside mask
    grid_idx = np.clip(grid_pts.astype(int), 0, shape - 1)
    inside = mask[grid_idx[:, 0], grid_idx[:, 1], grid_idx[:, 2]]
    interior_vox = grid_pts[inside]

    # Convert to MNI
    interior_mni = interior_vox + origin

    # Subsample if too many
    if len(interior_mni) > max_pts:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(interior_mni), size=max_pts, replace=False)
        interior_mni = interior_mni[idx]

    return interior_mni
