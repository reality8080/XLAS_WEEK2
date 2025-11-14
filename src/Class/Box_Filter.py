import numpy as np
from Class.Calculator import Calculator

class Box_Filter:
    def __init__(self):
        self.image = None
    @staticmethod
    def filter(img:np.ndarray, kernel_size:int, padding: bool = True)->np.ndarray:

        if len(img.shape) != 2:
            raise ValueError("Box_Filter chỉ hỗ trợ ảnh xám 2D.")
        
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("Kích thước kernel phải là số lẻ và >= 1.")

        kernel_matrix = np.ones((kernel_size,kernel_size), dtype=np.float32)
        kernel_matrix /=kernel_size*kernel_size
        
        return Calculator.convolution(img, kernel_matrix, padding=padding)