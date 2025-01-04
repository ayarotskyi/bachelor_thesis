import numpy as np
import cv2

class MemoryStack():
    MAX_STACK_SIZE = 4
    stack = None

    def __init__(self):
        self.stack = np.zeros((400, 400))

    def push(self, image):
        preprocessed_image = self.preprocess(image)
        self.stack = np.concatenate((preprocessed_image, self.stack[:300, :]))
        return self.stack

    def preprocess(self, image):
        image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (400, 200))[100:, :]
        blurred = cv2.GaussianBlur(image, (15, 15), 10)
        median_intensity = np.median(blurred)
        lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity))
        upper_threshold = int(min(255, (1.0 + 0.33) * median_intensity))
        canny_edges = cv2.Canny(blurred, 
                                 threshold1=lower_threshold, 
                                 threshold2=upper_threshold, 
                                 apertureSize=5)
        return canny_edges