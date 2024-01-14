import dxshot
import pyautogui
from utils.augmentations import letterbox
import numpy as np
import time

WIDTH, HEIGHT = pyautogui.size()
CENTER = [WIDTH/2, HEIGHT/2]
SIZE = 640
LEFT = int(CENTER[0] - SIZE / 2)
TOP = int(CENTER[1] - SIZE / 2)
REGION = (LEFT, TOP, LEFT+SIZE, TOP+SIZE)

class LoadScreen:
    def __init__(self, region: tuple[int,int,int,int]=REGION):
        self.region = region
        self.camera = dxshot.create(region=self.region, output_color="RGB")
        self.camera.start(target_fps=144, video_mode=True)
        
    def __iter__(self):
        return self

    def __next__(self):
        # now_time = time.time()
        
        im0 = self.camera.get_latest_frame()
        while im0 is None:
            im0 = self.camera.get_latest_frame()
        
        im = im0.copy()
        im = im.transpose((2, 0, 1))
        im = np.ascontiguousarray(im)
        
        # that_time = time.time()
        # print("Grab takes {:.2f} ms".format((that_time-now_time)*1E3))
        # print(im0)
        
        return im, im0
        