# -*- coding: utf-8 -*-
import threading
import time
import ctypes
import os
import pyautogui
import math

#Logitech import
try:
    root = os.path.abspath(os.path.dirname(__file__))
    driver = ctypes.CDLL(f'{root}/logitech.driver.dll')
    ok = driver.device_open() == 1
    if not ok:
        print('Error, GHUB or LGS driver not found')
except FileNotFoundError:
    print(f'Error, DLL file not found')


class Logitech:

    class mouse:

        # code: 1:左键, 2:中键, 3:右键

        @staticmethod
        def press(code):
            if not ok:
                return
            driver.mouse_down(code)

        @staticmethod
        def release(code):
            if not ok:
                return
            driver.mouse_up(code)

        @staticmethod
        def click(code):
            if not ok:
                return
            driver.mouse_down(code)
            driver.mouse_up(code)

        @staticmethod
        def scroll(a):
            if not ok:
                return
            driver.scroll(a)

        @staticmethod
        def move(x, y):
            # x: 水平移动的方向和距离, 正数向右, 负数向左
            # y: 垂直移动的方向和距离
            if not ok:
                return
            if x == 0 and y == 0:
                return
            driver.moveR(x, y, True)

    class keyboard:

        """
        键盘按键函数中，传入的参数采用的是键盘按键对应的键码
        code: 'a'-'z':A键-Z键, '0'-'9':0-9, 其他的没猜出来
        """

        @staticmethod
        def press(code):

            if not ok:
                return
            driver.key_down(code)

        @staticmethod
        def release(code):
            if not ok:
                return
            driver.key_up(code)

        @staticmethod
        def click(code):
            if not ok:
                return
            driver.key_down(code)
            driver.key_up(code)


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
DPI = 5
EDPI = 4000
SIZE = 640

print(CENTER)

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
            print(location)
            target = [(location[0] + location[2])/2 - SIZE/2, (location[1] + location[3])/2 - SIZE/2]
            weight = math.dist(target, [0, 0]) / math.dist([SIZE/2, SIZE/2], [0, 0])
            weight_pow = math.pow(weight, 2)
            x = target[0]
            y = target[1]
            if math.dist(target, [0, 0]) < 70:
                x *= math.sqrt(weight)
                y *= math.sqrt(weight)
            else:
                x *= weight_pow
                y *= weight_pow
            print(x, y)
            driver.moveR(int(x), int(y), True)
            
            location.clear()
            xyxy.clear()
        
        time.sleep(0.001)
     
     
     

monitor_thread = threading.Thread(target=monitor_global_var)
monitor_thread.start()


def run():
    monitor_thread = threading.Thread(target=monitor_global_var)
    monitor_thread.start()


def stop():
    monitor_thread.join()
    
