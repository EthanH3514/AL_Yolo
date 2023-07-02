import time
import cv2
import pyautogui
import dxshot
# import sys
# import numpy as np
# from PIL import Image
# import dxcam



WIDTH, HEIGHT = pyautogui.size()
CENTER = [WIDTH/2, HEIGHT/2]
SIZE = 640
LEFT = int(CENTER[0] - SIZE / 2)
TOP = int(CENTER[1] - SIZE / 2)
REGION = (LEFT, TOP, LEFT+SIZE, TOP+SIZE)


if __name__ == '__main__':
    img_count = 0
    Flag = True
    region = REGION
    camera = dxshot.create(region=region, output_color="BGR")
    camera.start(target_fps=144, video_mode=True)
    that_time = time.perf_counter()
    while Flag:
        img = camera.get_latest_frame()
        if img is None:
            continue
        img_count += 1
        this_time = time.perf_counter()
        duration_time = this_time - that_time
        if duration_time >= 1:
            that_time = this_time
            cv2.setWindowTitle('', str(img_count/duration_time))
            img_count = 0
            
        # sys.stdout.write("FPS is {:.2f}\n".format(img_count/duration_time))
        # sys.stdout.flush()
        
        # start = time.perf_counter()
        cv2.imshow('', img)
        
        # end = time.perf_counter()
        
        # sys.stdout.write("Show a image takes {:.2f}ms\n".format((end-start)*1E3))
        
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            Flag=False
            cv2.destroyAllWindows()
            break