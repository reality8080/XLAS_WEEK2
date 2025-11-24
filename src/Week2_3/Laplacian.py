import numpy as np
import cv2

class Laplacian:
    def __init__(self):
        self.image = None

    def Filter(self, image: np.ndarray, ksize=3, scale=0.3):
        if len(image.shape) == 3:
            result = np.zeros_like(image, dtype=np.float32)
            for c in range(3):
                lap = cv2.Laplacian(image[..., c].astype(np.float32), cv2.CV_32F, ksize=ksize)
                result[..., c] = image[..., c] - scale * lap
            return np.clip(result, 0, 255).astype(np.uint8)
        else:
            lap = cv2.Laplacian(image.astype(np.float32), cv2.CV_32F, ksize=ksize)
            return np.clip(image - scale * lap, 0, 255).astype(np.uint8)