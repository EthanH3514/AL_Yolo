import dxcam
import time
import cv2
import pyautogui
import numpy as np

camera = dxcam.create(output_color="BGR")

WIDTH, HEIGHT = pyautogui.size()
CENTER = [WIDTH/2, HEIGHT/2]
SIZE = 640
LEFT = int(CENTER[0] - SIZE / 2)
TOP = int(CENTER[1] - SIZE / 2)
REGION = (LEFT, TOP, LEFT+SIZE, TOP+SIZE)

if __name__ == '__main__':
    img_count = 0
    that_time = time.time()
    Flag = True
    region = REGION
    while(Flag):
        img = np.array(camera.grab(region=region))
        if img.any() == None:
            continue
        img_count += 1
        this_time = time.time()
        duration_time = this_time - that_time
        if duration_time >= 10:
            that_time = this_time
            img_count = 0
        cv2.imshow('', img)
        cv2.setWindowTitle('', str(img_count/duration_time))
        print(img_count)
        # print("FPS is {:.2f}".format(img_count/duration_time))
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            Flag=False
            cv2.destroyAllWindows()
            break