# batch_inference_classify.py — 批量判定 Good/Defect

import sys, csv
from pathlib import Path

from inference_classify import is_defect

def main(step_dir: str,
         model_path: str = 'pre_train.pth',
         out_csv: str = None):
    """
    step_dir  : 要检测的 .step 文件所在文件夹
    model_path: 分类模型文件，默认为 pointnet_defect.pth
    out_csv   : （可选）如果给出，则会把结果写入这个 CSV
    """
    d = Path(step_dir)
    files = sorted(d.glob('*.step'))
    if not files:
        print(f"❌ 在目录 {step_dir!r} 中没有找到任何 .step 文件")
        return

    writer = None
    if out_csv:
        f = open(out_csv, 'w', newline='')
        writer = csv.writer(f)
        writer.writerow(['filename', 'prediction'])

    for p in files:
        defect = is_defect(str(p), model_file=model_path)
        label  = 'DEFECT' if defect else 'GOOD'
        print(f"{p.name:30s} → {label}")
        if writer:
            writer.writerow([p.name, label])

    if writer:
        f.close()
        print(f"✅ 检测结果已保存到 {out_csv}")

if __name__=='__main__':
    if not (2 <= len(sys.argv) <= 4):
        print("Usage: python batch_inference_classify.py step_dir [model.pth] [out.csv]")
        sys.exit(1)

    step_dir   = sys.argv[1]
    model_pth  = sys.argv[2] if len(sys.argv)>=3 else 'pre_train.pth'
    out_csv    = sys.argv[3] if len(sys.argv)==4 else None

    main(step_dir, model_pth, out_csv)
