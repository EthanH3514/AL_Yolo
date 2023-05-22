import pyautogui
import numpy as np

screen_width, screen_height = pyautogui.size()
center_x = screen_width // 2
center_y = screen_height // 2

class Capture:
    def __init__(self, center=(center_x, center_y), size=((600, 600))):
        self.center = center
        self.size = size
        
    def work(self):
        screen = np.array(pyautogui.screenshot())
        x, y = self.center[0] - self.size[0] // 2, self.center[1] - self.size[1] // 2
        return screen[y:y+self.size[1], x:x+self.size[0]]
        
