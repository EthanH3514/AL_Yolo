# -*- coding: utf-8 -*-
from detect import YOLOv5Detector
import tkinter as tk
import threading
from pynput import keyboard


detector = YOLOv5Detector(
    weights='./weights/best.pt',
    data='./configs/AL_data.yaml',
    imgsz=(640, 640),
    conf_thres=0.45,
    iou_thres=0.45,
    enemy_label=0,
    device='cuda'
)

is_initiate = False
is_start_mouse = False

def enable_button(button):
    button.config(state=tk.NORMAL)


def initiate():
    global is_initiate
    if not is_initiate:
        detector_thread = threading.Thread(target=detector.work)
        detector_thread.start()
        is_initiate = True
        initiate_button.config(state=tk.DISABLED)
        root.after(1000, lambda: enable_button(initiate_button))
        print("目标检测已开启")

def end():
    global is_initiate
    if is_initiate:
        detector.stop()
        is_initiate = False
        end_button.config(state=tk.DISABLED)
        root.after(1000, lambda: enable_button(end_button))
        print("目标检测已停止")

def release():
    global is_initiate
    if is_initiate:
        detector.stop()
        is_initiate = False
    root.destroy()
    print("关闭程序")

def start_mouse():
    global is_start_mouse
    if not is_start_mouse:
        detector.start_mouse()
        is_start_mouse = True
        print("鼠标锁定已开启")

def stop_mouse():
    global is_start_mouse
    if is_start_mouse:
        detector.stop_mouse()
        is_start_mouse = False
        print("鼠标锁定已关闭")

def on_press(key):
    try:
        if key == keyboard.Key.page_up:  # 使用 PgUp 键开启鼠标锁定
            start_mouse()
        elif key == keyboard.Key.page_down:  # 使用 PgDn 键停止鼠标锁定
            stop_mouse()
    except Exception as e:
        print('Error:', e)
        
def listen_keyboard():
    # 创建键盘监听器，该监听器只响应按键事件
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

if __name__ == '__main__':
    root = tk.Tk()
    root.title("AL_Yolo")
    root.geometry("300x200")
    initiate_button = tk.Button(root, text="开启目标检测", command=initiate)
    initiate_button.grid(row=0, column=0)
    end_button = tk.Button(root, text="停止目标检测", command=end)
    end_button.grid(row=1, column=0)
    start_button = tk.Button(root, text="开启鼠标锁定", command=start_mouse)
    start_button.grid(row=0, column=1)
    stop_button = tk.Button(root, text="暂停鼠标锁定", command=stop_mouse)
    stop_button.grid(row=1, column=1)

    release_button = tk.Button(root, text="退出", command=release)
    release_button.grid(row=0, column=2)
    
    # 启动键盘监听线程
    listener_thread = threading.Thread(target=listen_keyboard, daemon=True)
    listener_thread.start()
    
    root.mainloop()
