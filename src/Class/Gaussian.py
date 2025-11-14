from Class.Calculator import Calculator
import numpy as np

class Gaussian:
    @staticmethod
    def gaussian_kernel(size: int, sigma: float)->np.ndarray:
        # Khởi tạo kernel rỗng
        kernel = np.zeros((size, size), dtype=np.float32)
        center = size//2
        for i in range(size):
            for j in range(size):
                x=i-center
                y=j-center
                kernel[i,j] = 1/(2*np.pi*np.pow(sigma,2))*np.exp(-(x**2+y**2)/2*sigma**2)

        kernel = kernel/np.sum(kernel)
        return kernel

    @staticmethod
    def filter(img:np.ndarray, kernel_size:int,sigma: float = 1.0, padding: bool = True)->np.ndarray:

        if len(img.shape) != 2:
            raise ValueError("Box_Filter chỉ hỗ trợ ảnh xám 2D.")
        
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("Kích thước kernel phải là số lẻ và >= 1.")

        kernel_matrix = Gaussian.gaussian_kernel(kernel_size, sigma)
        return Calculator.convolution(img, kernel_matrix, padding=padding)