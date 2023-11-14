import mouse_driver.ghub_mouse as ghub
import pyautogui

def ghub_mouse_move(rel_x, rel_y):
    ghub.mouse_xy(round(rel_x), round(rel_y))

def pygui_mouse_move(rel_x, rel_y):
    pyautogui.moveRel(rel_x, rel_y)

if __name__ == "__main__":
    import time
    trials = 10000
    start_time = time.time()
    for i in range(trials):
        ghub_mouse_move(1000,0)
        ghub_mouse_move(-1000,0)
    fps = trials/(time.time()-start_time)
    print(fps)