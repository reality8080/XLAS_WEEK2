import numpy as np
import cv2

class Histogram:
    def __init__(self):
        self.image = None

    def normalized_Histogram(self, img: np.ndarray):
        """Tính hist norm cho xám hoặc RGB (trung bình kênh)."""
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
        hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        return img_gray, hist.flatten() / hist.sum()

    def histogram_Equalization(self, img: np.ndarray):  # **Thay đổi: Input trực tiếp img**
        """Equalize cho xám hoặc RGB (CLAHE cho màu tốt hơn)."""
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_eq = clahe.apply(l)
            lab_eq = cv2.merge([l_eq, a, b])
            return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        else:
            # Xám: Dùng built-in
            return cv2.equalizeHist(img)