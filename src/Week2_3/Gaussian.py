import numpy as np
from Week2_3.Calculator import Calculator
import cv2  # **Thêm cho blur nhanh**

class Gaussian:
    @staticmethod
    def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
        """Vectorize kernel: Nhanh hơn vòng lặp."""
        center = size // 2
        x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)
        kernel = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        return kernel / np.sum(kernel)

    @staticmethod
    def filter(img: np.ndarray, kernel_size: int, sigma: float = 1.0, padding: bool = True) -> np.ndarray:
        """Ưu tiên cv2.GaussianBlur cho tốc độ."""
        if len(img.shape) != 2:
            raise ValueError("Gaussian chỉ hỗ trợ ảnh xám 2D.")
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("Kích thước kernel phải là số lẻ và >= 1.")
        
        # **cv2.GaussianBlur: Tối ưu nhất**
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma, borderType=cv2.BORDER_REPLICATE if padding else cv2.BORDER_CONSTANT)
        
        # Fallback: Calculator.convolution(img, Gaussian.gaussian_kernel(kernel_size, sigma), padding=padding)