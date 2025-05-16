# utils.py  ── STEP → sampled point cloud  (requires ocp 7.8+)
import numpy as np, random
from OCP.STEPControl import STEPControl_Reader
from OCP.BRepMesh    import BRepMesh_IncrementalMesh
from OCP.TopExp      import TopExp_Explorer
from OCP.TopAbs      import TopAbs_FACE
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.TopoDS import TopoDS
import random, numpy as np
from OCP.BRepMesh      import BRepMesh_IncrementalMesh
from OCP.TopExp        import TopExp_Explorer
from OCP.TopAbs        import TopAbs_FACE
from OCP.BRepAdaptor   import BRepAdaptor_Surface
from OCP.TopoDS        import TopoDS
# ── adaptive UV bounds (ocp & pythonocc-core) ──────────────────────────────
try:                                             # -------- ocp branch
    from OCP.BRepTools import BRepTools
    from OCP.TopoDS    import TopoDS            # <— 转型工具

    def uv_bounds(shape):
        """Return umin, umax, vmin, vmax for a TopoDS_Face (ocp ≥ 7.8)."""
        if hasattr(BRepTools, "UVBounds_s"):     # 优先用 _s(…) 版本
            face = TopoDS.Face_s(shape)         # 关键一步：Shape → Face
            return BRepTools.UVBounds_s(face)
        # fallback for极老ocp
        import ctypes
        u1=u2=v1=v2=ctypes.c_double()
        BRepTools.UVBounds(shape, u1, u2, v1, v2)
        return u1.value, u2.value, v1.value, v2.value

except ImportError:                              # ---- pythonocc-core
    from OCC.Core.BRepTools import breptools_UVBounds
    from OCC.Core.TopoDS   import topods_Face

    def uv_bounds(shape):
        return breptools_UVBounds(topods_Face(shape))
# ───────────────────────────────────────────────────────────────────────────

def load_step(path):
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != 1:
        raise RuntimeError(f"Cannot read {path}")
    reader.TransferRoots()
    return reader.OneShape()

def sample_shape(shape, n_pts=4000, seed=None):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    BRepMesh_IncrementalMesh(shape, 0.5)     # 0.5 mm deflection

    pts = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face_shp = exp.Current()
        face = TopoDS.Face_s(face_shp)  # ← 关键：Shape → Face
        u1,u2,v1,v2 = uv_bounds(face)
        surf = BRepAdaptor_Surface(face, True)
        for _ in range(50):                  # ≈50×faces, truncated later
            u = random.random()*(u2-u1)+u1
            v = random.random()*(v2-v1)+v1
            p  = surf.Value(u, v)
            pts.append([p.X(), p.Y(), p.Z()])
        exp.Next()

    pts = np.asarray(pts, dtype=np.float32)
    idx = np.random.choice(len(pts), n_pts, replace=len(pts) < n_pts)
    return pts[idx]
def normalise(pc: np.ndarray) -> np.ndarray:
    """Centre to origin and scale to unit sphere (no random jitter)."""
    pc = pc - pc.mean(0)
    pc = pc / np.linalg.norm(pc, axis=1).max()
    return pc

def sample_shape_with_faceids(shape, n_pts=4000, seed=None):
    """
    网格化 STEP，按面采点，返回 (pts, face_ids)：
    - pts: (n_pts,3) float32
    - face_ids: (n_pts,) int64
    """
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    # 网格化
    BRepMesh_IncrementalMesh(shape, 0.5)

    all_pts, all_fids = [], []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    fid = 0
    while exp.More():
        face_shp = exp.Current()
        face = TopoDS.Face_s(face_shp)
        # UV range
        u1,u2,v1,v2 = uv_bounds(face)
        surf = BRepAdaptor_Surface(face, True)
        # 每面均等采 M 点（M ~ n_pts / n_faces，向下取整）
        M = max(1, n_pts // max(1, shape.NbChildren()))  # 简易估算
        for _ in range(M):
            u = random.random()*(u2-u1)+u1
            v = random.random()*(v2-v1)+v1
            p = surf.Value(u,v)
            all_pts.append([p.X(), p.Y(), p.Z()])
            all_fids.append(fid)
        fid += 1
        exp.Next()

    pts      = np.array(all_pts, dtype=np.float32)
    face_ids = np.array(all_fids, dtype=np.int64)
    # 下采样到 n_pts
    if len(pts) >= n_pts:
        idx = np.random.choice(len(pts), n_pts, replace=False)
    else:
        idx = np.random.choice(len(pts), n_pts, replace=True)
    return pts[idx], face_ids[idx]