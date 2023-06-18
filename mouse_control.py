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
SIZE = 640
SMOOTH = 1.2

print(CENTER, (WIDTH, HEIGHT))

def get_target(location):
    vector = [(location[0] + location[2]) / 2 - SIZE / 2, (location[1] + location[3]) / 2 - SIZE / 2]
    return vector

def get_vector(target):
    for i in range(2):
        target[i] *= SMOOTH / MOUSE_SENSITIVITY
    return target

# quit_signal
quit_signal = True

def monitor_global_var():
    global Q_quit
    
    location = []
    
    global quit_signal
    while quit_signal:
        
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
                
            target = get_target(location)
            
            if abs(target[0]) < 10 and abs(target[1]) < 10:
                xyxy.clear()
                location.clear()
                time.sleep(0.001)
                continue
            
            vector = get_vector(target)


            print(target)
            mouse_move(vector[0], vector[1])
            
            location.clear()
            xyxy.clear()
        
        time.sleep(0.001)


monitor_thread = threading.Thread(target=monitor_global_var)

def run():
    monitor_thread.start()


def stop():
    global quit_signal
    quit_signal = False
    monitor_thread.join()
    