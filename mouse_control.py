import threading
import time
import pyautogui
from mouse_driver.MouseMove import mouse_move


# Global variable
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

# Quit signal
quit_signal = True

def monitor_global_var():
    
    location = []
    
    global quit_signal
    while quit_signal:
        
        xyxy = get_global_var()
            
        if len(xyxy) < 4:
            xyxy.clear()
            
        elif xyxy !=[]:
            for item in xyxy:
                location.append(float(item))
            if len(location) >= 4 and abs(location[3] - location[1]) >= 10:
                target = get_target(location)
                
                if abs(target[0]) >= 10 and abs(target[1]) >= 10:
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
    