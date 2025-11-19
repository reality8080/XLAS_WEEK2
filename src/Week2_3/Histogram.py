import numpy as np
import cv2


class Histogram:
    def __init__(self):
        self.image = None

    def normalized_Histogram(self, img: np.ndarray):

        if len(img.shape) == 3 and img.shape[2] == 3:
            # Nếu là ảnh màu, chuyển về ảnh xám để tính histogram 1D
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            # Nếu đã là ảnh xám (1 kênh), giữ nguyên
            img_gray = img
        hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        hist_norm = hist / hist.sum()
        return img_gray, hist_norm

    def histogram_Equalization(self,img_gray:np.ndarray, hist_norm):
        img_gray = img_gray.astype(np.uint8)
        cdf = np.cumsum(hist_norm)
        sk = np.round(255 * cdf).astype('uint8')
        img_eq = sk[img_gray]
        return img_eq
