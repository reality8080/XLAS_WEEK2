import numpy as np
from Week2_3.Calculator import Calculator
class Meadian_Filter:
    def __init__(self):
        self.image = None
    def Filter(self, image:np.ndarray, kernel_size, padding = True):

        if len(image.shape) != 2:
            raise ValueError("Meadian_Filter chỉ hỗ trợ ảnh xám 2D.")
        
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("Kích thước kernel phải là số lẻ và >= 1.")
        
        image=image.astype(np.float32)
        if padding:
            pad_h = pad_w = kernel_size//2

            image = np.pad(image, ((pad_h,pad_h), (pad_w,pad_w)), mode = 'constant',constant_values=0)
        h_image,w_image = image.shape

        B = np.zeros((h_image-kernel_size+1,w_image-kernel_size+1),dtype=np.float32)

        for i in range(0,h_image-kernel_size+1):
            for j in range(0,w_image-kernel_size+1):
                s_A = image[i:i+kernel_size, j:j+kernel_size]
                B[i,j] = np.median(s_A)

        return np.clip(B, 0, 255).astype(np.uint8)