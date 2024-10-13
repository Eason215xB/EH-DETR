import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('pt') # select your model.pt path
    model.predict(source='images/test',
                  project='runs/detect',
                  name='exp',
                  save=True,
                #   visualize=True # visualize model features maps
                  )