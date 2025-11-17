import numpy as np
from Class.Calculator import Calculator
import cv2

class Laplacian:
    def __init__(self):
        self.image = None
    def Filter(self, image:np.ndarray, kernel_size=3, padding = True, neighborhood=4):

        if len(image.shape) != 2:
            raise ValueError("Laplacian chỉ hỗ trợ ảnh xám 2D.")
        
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("Kích thước kernel phải là số lẻ và >= 1.")
        
        if neighborhood>kernel_size*kernel_size -1:
            raise ValueError("Số điểm lân cận không hợp lệ.")
        
        image=image.astype(np.float32)
        

        if padding:
            pad_h = pad_w = kernel_size//2

            image = np.pad(image, ((pad_h,pad_h), (pad_w,pad_w)), mode = 'constant',constant_values=0)

        # h_image,w_image = image.shape
        kernel_matrix = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        center = kernel_size // 2

        coords = []

        if neighborhood ==4:
            coords = [
                (center - 1, center),  # Up
                (center + 1, center),  # Down
                (center, center - 1),  # Left
                (center, center + 1)   # Right
            ]
        elif neighborhood ==8:
            coords = [
                (center - 1, center),  # Up
                (center + 1, center),  # Down
                (center, center - 1),  # Left
                (center, center + 1)   # Right
                (center - 1, center - 1),  # Top-Left
                (center - 1, center + 1),  # Top-Right
                (center + 1, center - 1),  # Bottom-Left
                (center + 1, center + 1)   # Bottom-Right
            ]
        elif neighborhood ==16:
            for i in range(kernel_size):
                for j in range(kernel_size):
                    if abs(i-center) + abs(j-center) <= kernel_size//2 and (i,j)!=(center,center):
                        coords.append((i,j))
        else:
            # Full (mọi neighbor trong kernel)
            coords = [(i,j) for i in range(kernel_size) for j in range(kernel_size) if (i,j)!=(center,center)]
            # Nếu neighbors nhỏ hơn tổng số, chọn subset đầu tiên
            coords = coords[:neighborhood]
        
        for (i,j) in coords:
            kernel_matrix[i,j] = -1
        
        kernel_matrix[center, center] = len(coords)

        B = Calculator.convolution(image, kernel_matrix, padding=True)

        result = image[:B.shape[0], :B.shape[1]] - B
        
        return np.clip(result, 0, 255).astype(np.uint8)
        