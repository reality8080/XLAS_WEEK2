import numpy as np
import cv2

class Piecewise_Linear_Transform:
    def __init__(self):
        self.image = None

    def auto_slicing_threshold(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0,256]).flatten()
        cumsum = np.cumsum(hist)

        total = np.sum(hist)

        A=np.searchsorted(cumsum, total*0.3)
        B=np.searchsorted(cumsum, total*0.7)

        return A,B

    def piecewise_linear_transform(self, img, r1 = 70, s1 = 0, r2 = 140, s2 = 255):
        # Bien doi kieu du lieu (int->float) tren gia tri anh
        img = img.astype(np.float32)
        # Tao mang gia tri pixel = 0 cung kich thuoc voi Anh
        result = np.zeros_like(img)

        # Cho 1 anh qua 3 bo loc
        # Chon cac gia tri vung toi
        mask1 = img<r1
        # Chon cac gia tri vung trung binh
        mask2 = (img>=r1) & (img <r2)
        # Chon cac gia tri vung sang
        mask3 = img >= r2
        # Ap dung gia tri voi tung bo loc khac nhau
        result[mask1] = (s1 / r1) * img[mask1]
        result[mask2] = ((s2 - s1)/ (r2 - r1)) * (img[mask2] - r1) + s1
        result[mask3] = ((255 - s2) / (255 - r2)) * (img[mask3]-r2) + s2
        # Gioi han va lam tron gia tritrong khoang 255
        # Chuyen ve uint8
        return np.uint8(np.clip(result,0,255))