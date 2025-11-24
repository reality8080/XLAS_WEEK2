import numpy as np
from scipy.ndimage import median_filter  # **Thêm SciPy**

class Median_Filter:  # **Sửa tên**
    # Ví dụ: Median_Filter.py
    @staticmethod
    def Filter(image: np.ndarray, kernel_size: int, padding: bool = True) -> np.ndarray:
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("Kích thước kernel phải là số lẻ và >= 1.")
        
        if len(image.shape) == 3:
            # Xử lý từng kênh
            result = np.stack([
                median_filter(image[..., c], size=kernel_size, mode='constant' if padding else 'nearest')
                for c in range(image.shape[2])
            ], axis=-1)
            return result.astype(np.uint8)
        
        # Ảnh xám
        return median_filter(image, size=kernel_size, mode='constant' if padding else 'nearest').astype(np.uint8)