# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from detect import YOLOv5Detector
from Capturer import Capture
import time

from pynput import mouse
from pynput import keyboard
import winsound

# capture = Capture()

capture_rate = 60
interval = 1 / capture_rate
start_time = time.time()

detector = YOLOv5Detector(
    weights='best.pt',
    data='AL_data.yaml',
    imgsz=(640, 640),
    conf_thres=0.45,
    iou_thres=0.4,
    device='cuda'
)

if __name__ == '__main__':
    winsound.Beep(800, 200)
    
    while(True):
        current_time = time.time()
        
        elapsed_time = current_time - start_time
        
        # capture_start_time = time.time()
        # image = capture.work()
        # capture_end_time = time.time()
        # capture_execution_time = capture_end_time - capture_start_time
        
        # print("Capturing used {:.3f} seconds".format(capture_execution_time))
        # 截图大约 27 ms
        
        if elapsed_time >=interval:
            # #region = capture.work()
            
            start_time = time.time()
            
            # cv2.imwrite('image_cache/region_cache.png', region)
            
            xywh = detector.work()
            
            # r, g, b = cv2.split(region)
            # #region = cv2.merge([b, g, r])
            # cv2.imshow('Center Region', region)
            # cv2.waitKey(2)

        time.sleep(0.001)



# def on_mouse_click(x, y, button, pressed):
#     if button == mouse.Button.x1:
#         if pressed:
#             # 侧键1按下
#             run()


# listener = mouse.Listener(on_click=on_mouse_click)
# listener.start() 


