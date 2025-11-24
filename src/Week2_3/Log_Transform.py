import numpy as np

class Log_Transform:
    def __init__(self):
        self.image = None

    def auto_c(self, img: np.ndarray):
        return 255 / np.log(1 + np.max(img))

    def log_transform(self, img: np.ndarray, c: float = None, flag: bool = True):
        img_float = img.astype(np.float32)
        if flag:
            c = self.auto_c(img_float)
        log_img = c * np.log(1 + img_float)  # Vectorize
        return np.uint8(np.clip(log_img, 0, 255))