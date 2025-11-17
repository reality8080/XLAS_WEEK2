# Week2_3/FrequencyFilters/Butterworth.py
import numpy as np
from Week2_3.Calculator import Calculator

class Butterworth:
    @staticmethod
    def create_filter(H: int, W: int, cutoff: float, order: int = 1, type: str = 'low') -> np.ndarray:
        u = np.arange(W) - W // 2
        v = np.arange(H) - H // 2
        U, V = np.meshgrid(u, v)
        D = np.sqrt(U**2 + V**2)

        D0 = cutoff
        if type == 'low':
            H = 1 / (1 + (D / D0)**(2 * order))
        elif type == 'high':
            H = 1 / (1 + (D0 / D)**(2 * order))
        else:
            raise ValueError("type must be 'low' or 'high'")

        return H.astype(np.float32)

    @staticmethod
    def filter(img: np.ndarray, cutoff: float, order: int = 1, type: str = 'low', padding: bool = True) -> np.ndarray:
        if len(img.shape) != 2:
            raise ValueError("Chỉ hỗ trợ ảnh xám 2D.")

        H, W = img.shape
        f = np.fft.fft2(img.astype(np.float64))
        fshift = np.fft.fftshift(f)

        filter_mask = Butterworth.create_filter(H, W, cutoff, order, type)
        fshift_filtered = fshift * filter_mask

        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        return np.clip(img_back, 0, 255).astype(np.uint8)