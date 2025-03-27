import numpy as np
import cv2


class MemoryStack:
    stack: np.ndarray = None
    history: np.ndarray = None

    def __init__(self, stack_size=4):
        self.stack = np.zeros((stack_size, 100, 200))
        self.size = 0
        self.history = np.array([[0, 0]] * stack_size)

    def push(self, image):
        preprocessed_image = self.preprocess(image)
        self.stack = np.concatenate([self.stack[1:], [preprocessed_image]])
        return self.stack

    def push_history(self, prediction):
        self.history = np.concatenate([self.history[1:], [prediction]])
        return self.history

    @staticmethod
    def preprocess(image):
        image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (200, 200))[100:, :]
        blurred = cv2.GaussianBlur(image, (15, 15), 10)
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
        sobel_combined = cv2.magnitude(sobel_x, sobel_y)
        sobel_normalized = cv2.normalize(
            sobel_combined, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        return sobel_normalized / 175.0 - 1


class MemoryStackDAVE2(MemoryStack):
    def __init__(self, stack_size=3):
        self.stack = np.zeros((stack_size, 22, 200, 3))
        self.size = 0
        self.history = np.array([[0, 0]] * stack_size)

    @staticmethod
    def preprocess(image):
        image = cv2.resize(image, (200, 200))[100:122, :]
        return image
