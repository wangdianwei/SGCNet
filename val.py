# val_like_train.py
import warnings
warnings.filterwarnings("ignore")

from ultralytics import YOLO
from ultralytics import RTDETR

if __name__ == "__main__":
    # ① 训练出来的权重（best/last）
    WEIGHTS = r""
    # ② 数据集配置（与训练一致）
    DATA = r"l"

    model = YOLO(WEIGHTS)

    # ③ 原生验证（会自动打印与 train 一样的表格）
    model.val(
        data=DATA,
        split="test",
        imgsz=640,
        batch=2,
        device="0",
        # conf=0.01,
        # iou=0.6,
        max_det=300,
        half=False,
        verbose=True,      # 关键：打印 per-class 表格
        plots=True,       # 需要图再开 True
        save_json=False,   # 需要 COCO json 再开 True
        project=r"",
        name="",
    )
