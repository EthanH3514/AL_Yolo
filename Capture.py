import dxshot
import pyautogui
from utils.augmentations import letterbox
import numpy as np
import time
import threading

WIDTH, HEIGHT = pyautogui.size()
CENTER = [WIDTH/2, HEIGHT/2]
SIZE = 640
LEFT = int(CENTER[0] - SIZE / 2)
TOP = int(CENTER[1] - SIZE / 2)
REGION = (LEFT, TOP, LEFT+SIZE, TOP+SIZE)

data = None

class LoadScreen:
    def __init__(self, img_size = SIZE, stride = 32, auto=True, transforms=None, region: tuple[int,int,int,int]=REGION):
        self.screen = 0
        self.img_size = img_size
        self.region = region
        self.camera = dxshot.create(region=REGION, output_color="BGR")
        self.frame = 0
        self.stride = stride
        self.auto = auto
        self.transforms = transforms
        self.left = region[0]
        self.top = region[1]
        self.width = img_size
        self.height = img_size
        
        self.camera.start(target_fps=144, video_mode=True)
        
    def __iter__(self):
        return self

    def __next__(self):
        # now_time = time.time()
        
        im0 = self.camera.get_latest_frame()
        while im0 is None:
            time.sleep(0.0005)
            im0 = self.camera.get_latest_frame()
        
        s = f'screen {self.screen} (LTWH): {self.left},{self.top},{self.width},{self.height}: '
        if self.transforms:
            im = self.transforms(im0)
        else:
            im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]
            im = im.transpose((2, 0, 1))[::-1]
            im = np.ascontiguousarray(im)
        self.frame += 1
        
        # that_time = time.time()
        # print("Grab takes {:.2f} ms".format((that_time-now_time)*1E3))
        # print(im0)
        
        return str(self.screen), im, im0, None, s
        