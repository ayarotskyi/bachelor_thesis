import numpy as np
import cv2

class MemoryStack():
    size = None
    stack: np.ndarray = None
    max_size = None

    def __init__(self, stack_size = 4):
        self.stack = np.zeros((stack_size, 100, 400))
        self.max_size = stack_size
        self.size = 0

    def push(self, image):
        preprocessed_image = self.preprocess(image)
        self.stack = np.concatenate([self.stack[1:], [preprocessed_image]])
        if self.size < self.stack.shape[0]:
            self.size += 1
        return self.stack

    @staticmethod
    def preprocess(image):
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