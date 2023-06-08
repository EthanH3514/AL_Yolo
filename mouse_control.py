# -*- coding: utf-8 -*-
import threading
import time
import os
import pyautogui
import math
from pynput import mouse, keyboard
from ctypes import CDLL
import numpy as np
from mouse_driver.MouseMove import mouse_move


# 全局变量传送
global_xyxy = []

lock = threading.Lock()


def set_global_var(value):
    global global_xyxy
    with lock:
        global_xyxy = value

def get_global_var():
    global global_xyxy
    with lock:
        return global_xyxy
    

WIDTH, HEIGHT = pyautogui.size()
CENTER = [WIDTH/2, HEIGHT/2]
FOV = 110
DPI = 800
MOUSE_SENSITIVITY = 5
EDPI = DPI * MOUSE_SENSITIVITY
SIZE = 640
D = math.sqrt(WIDTH * WIDTH + HEIGHT * HEIGHT) / 2 / math.tan(FOV*math.pi/360)
SMOOTH = 0.5

print(CENTER, (WIDTH, HEIGHT))

def get_vector(location):
    vector = [((location[0] + location[2]) / 2 - SIZE / 2) / MOUSE_SENSITIVITY, ((location[1] + location[3]) / 2 - SIZE / 2) / MOUSE_SENSITIVITY]
    return vector

def monitor_global_var():
    location = []
    while True:
        xyxy = get_global_var()
        if len(xyxy) < 4:
            xyxy.clear()
            time.sleep(0.001)
            continue
        if xyxy !=[]:
            for item in xyxy:
                location.append(float(item))
            if len(location) < 4:
                xyxy.clear()
                location.clear()
                time.sleep(0.001)
                continue
            if abs(location[3] - location[1]) < 10:
                xyxy.clear()
                location.clear()
                time.sleep(0.001)
            if location[0] < SIZE/2+2 and location[2] >SIZE/2-2 and location[1] < SIZE/2+2 and location[3] > SIZE/2-2:
                xyxy.clear()
                location.clear()
                time.sleep(0.001)
                continue
            # print(location)
            target = get_vector(location)
            print(target)
            mouse_move(target[0], target[1])
            
            location.clear()
            xyxy.clear()
        
        time.sleep(0.003)
     
     
     

monitor_thread = threading.Thread(target=monitor_global_var)
monitor_thread.start()


def run():
    monitor_thread = threading.Thread(target=monitor_global_var)
    monitor_thread.start()


def stop():
    monitor_thread.join()
    
