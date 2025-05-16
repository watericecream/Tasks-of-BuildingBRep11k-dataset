import pathlib, numpy as np
from utils import load_step, sample_shape_with_faceids, normalise
from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCP.BRepTools import BRepTools_ReShape

# 1. 原始完整模型目录
SRC_DIR = pathlib.Path('data/good')    # 文件名格式："123.step"
# 2. 输出带缺陷的 STEP
OUT_STEP = pathlib.Path('data/bad')
OUT_STEP.mkdir(exist_ok=True)
# 3. 输出对应的点云标签 (pts, mask)
OUT_LBL  = pathlib.Path('data/bad_labels')
OUT_LBL.mkdir(exist_ok=True)

# 缺陷生成规则示例
# 例如：删除面索引列表（此处填你的删除规则）
def select_del_fids(shape):
    # 返回要删除的 face id 列表
    # TODO: 填写你的选择逻辑，例如随机删除1-3个面的索引
    import random
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    # 先统计面数量
    exp_count = TopExp_Explorer(shape, TopAbs_FACE)
    count = 0
    while exp_count.More():
        count += 1
        exp_count.Next()
    # 随机选择一个或多个面进行删除，这里示例删除一个面
    face_ids = list(range(count))
    return random.sample(face_ids, k=1)

for good_path in sorted(SRC_DIR.glob('*.step')):
    name = good_path.stem  # 例如 '123'
    # 加载完整模型
    good_shape = load_step(good_path)

    # 复制并生成缺陷模型
    del_fids = select_del_fids(good_shape)
    reshaper = BRepTools_ReShape()
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS

    # 遍历 face，根据索引删除
    exp = TopExp_Explorer(good_shape, TopAbs_FACE)
    fid = 0
    while exp.More():
        if fid in del_fids:
            face_shp = exp.Current()
            face = TopoDS.Face_s(face_shp)
            reshaper.Remove(face)
        fid += 1
        exp.Next()
    bad_shape = reshaper.Apply(good_shape)

    # 写入坏的 STEP
    bad_step_path = OUT_STEP / f"{name}_def.step"
    writer = STEPControl_Writer()
    writer.Transfer(bad_shape, STEPControl_AsIs)
    writer.Write(str(bad_step_path))

    # 采样 pts 和 face_ids
    pts, fids = sample_shape_with_faceids(bad_shape, n_pts=4000)
    mask = np.isin(fids, del_fids).astype(np.int64)

    # 保存点云标签
    np.savez_compressed(
        OUT_LBL / f"{name}.npz",
        pts=pts.astype(np.float32),
        mask=mask
    )

    print(f"Processed {name}: deleted faces {del_fids}, bad STEP → {bad_step_path}")
