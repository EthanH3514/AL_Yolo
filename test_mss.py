import mss
import time
import numpy as np
import cv2


class LoadScreenshots:
    # YOLOv5 screenshot dataloader, i.e. `python detect.py --source "screen 0 100 100 512 256"`
    def __init__(self, img_size=640, stride=32, auto=True, transforms=None):
        # source = [screen_number left top width height] (pixels)
        source = "center 640 220 640 640"
        import mss

        source, *params = source.split()
        self.screen, left, top, width, height = 0, None, None, None, None  # default to full screen 0
        if len(params) == 1:
            self.screen = int(params[0])
        elif len(params) == 4:
            left, top, width, height = (int(x) for x in params)
        elif len(params) == 5:
            self.screen, left, top, width, height = (int(x) for x in params)
        self.img_size = img_size
        self.stride = stride
        self.transforms = transforms
        self.auto = auto
        self.mode = 'stream'
        self.frame = 0
        self.sct = mss.mss()

        # Parse monitor shape
        monitor = self.sct.monitors[self.screen]
        self.top = monitor['top'] if top is None else (monitor['top'] + top)
        self.left = monitor['left'] if left is None else (monitor['left'] + left)
        self.width = width or monitor['width']
        self.height = height or monitor['height']
        self.monitor = {'left': self.left, 'top': self.top, 'width': self.width, 'height': self.height}

    def __iter__(self):
        return self

    def __next__(self):
        im0 = np.array(self.sct.grab(self.monitor))[:, :, :3]  # [:, :, :3] BGRA to BGR
        
        return im0


if __name__ == '__main__':
    sct = LoadScreenshots()
    Flag = True
    that_time = time.time()
    img_count = 0
    while Flag:
        img = sct.__next__()
        if img is None:
            continue
        img_count += 1
        this_time = time.time()
        duration_time = this_time - that_time
        if duration_time >= 10:
            that_time = this_time
            img_count = 0
        cv2.imshow('', img)
        cv2.setWindowTitle('', str(img_count/duration_time))
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            Flag=False
            cv2.destroyAllWindows()
            break
        