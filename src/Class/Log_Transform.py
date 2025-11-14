import numpy as np

class Log_Transform:
    def __init__(self):
        self.image = None

    def auto_c(self,img:np.array):
        max_array=np.max(img)
        c = 255 / np.log(1 + max_array)
        return c

    def log_transform(self,img,c, flag=True):
        # Bien doi kieu du lieu (int->float) tren gia tri anh
        img_float = img.astype(np.float32)

        # Tim gia tri pixel_max tren anh
        if flag:
            c = self.auto_c(img_float)
        # Tinh toan gia tri c theo gia tri pixel_max
        # Dam bao c trong mien gia tri [0,255]

        # Dung log bien doi hinh anh theo gia tri c va log
        log_img = c * np.log(1+img_float)

        # Lam tron và gioi han trong khoang [0, 255]
        # Chuyen ve gia tri uint8 cua tung pixel
        return np.uint8(np.clip(log_img,0,255))