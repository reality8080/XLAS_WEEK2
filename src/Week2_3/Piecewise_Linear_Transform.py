import numpy as np
import cv2

class Piecewise_Linear_Transform:
    def __init__(self):
        self.image = None

    def auto_slicing_threshold(self, img):
        """Auto r1, r2 dùng Otsu (tốt hơn cumsum)."""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        _, r1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        r2 = 255 - r1  # Symmetric
        return int(r1 * 0.5), int(r2 * 0.5)  # Adjust for piecewise

    def piecewise_linear_transform(self, img, r1=70, s1=0, r2=140, s2=255):
        img = img.astype(np.float32)
        result = np.zeros_like(img)
        mask1 = img < r1
        mask2 = (img >= r1) & (img < r2)
        mask3 = img >= r2
        result[mask1] = (s1 / r1) * img[mask1]
        result[mask2] = ((s2 - s1) / (r2 - r1)) * (img[mask2] - r1) + s1
        result[mask3] = ((255 - s2) / (255 - r2)) * (img[mask3] - r2) + s2
        return np.uint8(np.clip(result, 0, 255))