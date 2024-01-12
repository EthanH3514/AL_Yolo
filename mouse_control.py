# import threading
# import time
import pyautogui
from mouse_driver.MouseMove import ghub_mouse_move
import numpy as np
import torch
# from mouse_driver.MouseMove import pygui_mouse_move as mouse_move


WIDTH, HEIGHT = pyautogui.size()
CENTER = [WIDTH/2, HEIGHT/2]
# FOV = 110
# DPI = 800
MOUSE_SENSITIVITY = 5
SIZE = 640
SMOOTH = 1.2

OFFSET = torch.tensor([SIZE / 2, SIZE / 2], device='cuda:0')

MUL = 2 / MOUSE_SENSITIVITY

print(CENTER, (WIDTH, HEIGHT))

def move_to(xyxy):

    if len(xyxy) >= 4:
        # stacked_array = np.stack([tensor.cpu().numpy() for tensor in xyxy])
        # target = stacked_array.flatten()
        top_left = torch.stack(xyxy[:2])
        bottom_right = torch.stack(xyxy[2:])
        # print("top_left is ", top_left)
        # print("bottom_right is ", bottom_right)
        target = ((top_left + bottom_right) / 2 - OFFSET) * MUL

        # print("target is ", target)

        ghub_mouse_move(target[0].item(), target[1].item())
        # ghub_mouse_move(x / MOUSE_SENSITIVITY * 2, y / MOUSE_SENSITIVITY * 2)
        
        # location.clear()
