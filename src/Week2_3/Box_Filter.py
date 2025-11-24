import numpy as np
from Week2_3.Calculator import Calculator

class Box_Filter:
    def __init__(self):
        self.image = None
    @staticmethod
    def filter(img:np.ndarray, kernel_size:int, padding: bool = True)->np.ndarray:

        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("Kích thước kernel phải là số lẻ và >= 1.")

        kernel_matrix = np.ones((kernel_size,kernel_size), dtype=np.float32)/(kernel_size**2)

        if len(img.shape) == 3:
            channels = [Calculator.convolution(img[:,:,c], kernel_matrix,padding=padding)for c in range(3)]
            return np.stack(channels, axis=2).astype(np.uint8)

        return Calculator.convolution(img, kernel_matrix, padding=padding)