# -*- coding: utf-8 -*-
from detect import YOLOv5Detector
import pynput
import winsound
import mouse_control


detector = YOLOv5Detector(
    weights='best.pt',
    data='AL_data.yaml',
    imgsz=(640, 640),
    conf_thres=0.45,
    iou_thres=0.4,
    device='cuda'
)

def release(key):
    if key == pynput.keyboard.Key.home:  # Home
        winsound.Beep(400, 200)
        mouse_control.run()
        return False
    elif key == pynput.keyboard.Key.end:  # End
        winsound.Beep(600, 200)
        mouse_control.stop()
        return False

if __name__ == '__main__':
    winsound.Beep(800, 200)
    
    listener = pynput.keyboard.Listener(on_release=release)
    listener.start()
    listener.join()
    
    detector.work()
    
    listener = pynput.keyboard.Listener(on_release=release)
    listener.start()
    listener.join()

