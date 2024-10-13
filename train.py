import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('FH-DETR.yaml')
    # model.load('') # loading pretrain weights
    model.train(data=r'data.yaml',
                cache=False,
                imgsz=640,
                epochs=1,
                batch=1,
                workers=1,
                device='0',
                project='runs/train',
                name='exp',
                )