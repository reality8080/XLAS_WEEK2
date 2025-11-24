import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, median_filter  # **Thêm SciPy**

class Max_Filter:
    # Max_Filter.py
    @staticmethod
    def Filter(image: np.ndarray, kernel_size: int, padding: bool = True) -> np.ndarray:
        size = (kernel_size, kernel_size)
        mode = 'constant' if padding else 'nearest'
        if len(image.shape) == 3:
            return np.stack([maximum_filter(image[..., c], size=size, mode=mode) 
                            for c in range(image.shape[2])], axis=-1).astype(np.uint8)
        return maximum_filter(image, size=size, mode=mode).astype(np.uint8)

class Min_Filter:
    @staticmethod
    def Filter(image: np.ndarray, kernel_size: int, padding: bool = True) -> np.ndarray:
        size = (kernel_size, kernel_size)
        mode = 'constant' if padding else 'nearest'
        if len(image.shape) == 3:
            return np.stack([minimum_filter(image[..., c], size=size, mode=mode) 
                            for c in range(image.shape[2])], axis=-1).astype(np.uint8)
        return minimum_filter(image, size=size, mode=mode).astype(np.uint8)

class Mid_Filter:  # **Đổi tên thành Average_MinMax_Filter nếu không phải median**
    @staticmethod
    def Filter(image: np.ndarray, kernel_size: int, padding: bool = True) -> np.ndarray:
        size = (kernel_size, kernel_size)
        mode = 'constant' if padding else 'nearest'
        if len(image.shape) == 3:
            return np.stack([((maximum_filter(image[..., c], size=size, mode=mode) + minimum_filter(image[..., c], size=size, mode=mode)) / 2)
                            for c in range(image.shape[2])], axis=-1).astype(np.uint8) 
        return ((maximum_filter(image, size=size, mode=mode) + minimum_filter(image, size=size, mode=mode)) / 2).astype(np.uint8)