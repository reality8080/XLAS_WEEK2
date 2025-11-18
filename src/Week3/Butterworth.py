# Week2_3/FrequencyFilters/Butterworth.py
import numpy as np
import time

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
    def filter(img: np.ndarray, cutoff: float, order: int = 1, type: str = 'low', verbose: bool = False) -> np.ndarray:
        if len(img.shape) != 2:
            raise ValueError("Chỉ hỗ trợ ảnh xám 2D.")

        H, W = img.shape
        t0 = time.perf_counter()

        t1 = time.perf_counter()
        f = np.fft.fft2(img.astype(np.float64))
        t2 = time.perf_counter()

        fshift = np.fft.fftshift(f)
        t3 = time.perf_counter()

        filter_mask = Butterworth.create_filter(H, W, cutoff, order, type)
        t4 = time.perf_counter()

        fshift_filtered = fshift * filter_mask
        t5 = time.perf_counter()

        f_ishift = np.fft.ifftshift(fshift_filtered)
        t6 = time.perf_counter()

        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        t7 = time.perf_counter()

        result = np.clip(img_back, 0, 255).astype(np.uint8)

        steps = {
            "1. FFT2":         t2 - t1,
            "2. FFTShift":     t3 - t2,
            "3. Tạo bộ lọc":   t4 - t3,
            "4. Nhân bộ lọc":  t5 - t4,
            "5. IFFTShift":    t6 - t5,
            "6. IFFT2":        t7 - t6,
            "Tổng thời gian":  t7-t0
        }
            

        return result, steps