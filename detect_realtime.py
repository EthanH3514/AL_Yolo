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
        
        if elapsed_time >=interval:
            
            start_time = time.time()
            
            xywh = detector.work()

        time.sleep(0.001)
