import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/SGCNet.yaml')
    # model.load('sgcl_mamba_p4_best.pt') # loading pretrain weights
    model.train(data=R'ultralytics/cfg/datasets/road_UAV_PDD_offical.yaml',
                cache=False,
                imgsz=640,
                epochs=400,
                batch=16,
                close_mosaic=5,
                workers=0,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                amp=False, # close amp
                # fraction=0.2,
                project='',
                name='',
                patience=0,
                # iou=0.45,
                )