import time
import dxcam
import pyautogui
import cv2

WIDTH, HEIGHT = pyautogui.size()
CENTER = [WIDTH/2, HEIGHT/2]
SIZE = 640
LEFT = int(CENTER[0] - SIZE / 2)
TOP = int(CENTER[1] - SIZE / 2)
REGION = (LEFT, TOP, LEFT+SIZE, TOP+SIZE)

start_time, fps = time.perf_counter(), 0
cam = dxcam.create(region=REGION, output_color="BGR")
# start = time.perf_counter()
while fps < 200:
    frame = cam.grab()
    if frame is None:
        continue
    # if frame is not None:  # New frame
    fps += 1
    cv2.imshow('p', frame)
        # print(fps)

end_time = time.perf_counter() - start_time
print("FPS is {:.2f}".format(fps/end_time))
key = cv2.waitKey(0)
if key == ord('q'):
    cv2.destroyAllWindows()