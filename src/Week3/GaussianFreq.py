# Week2_3/FrequencyFilters/GaussianFreq.py
import numpy as np
import time

class GaussianFreq:
    @staticmethod
    def create_filter(H: int, W: int, cutoff: float, type: str = 'low') -> np.ndarray:
        u = np.arange(W) - W // 2
        v = np.arange(H) - H // 2
        U, V = np.meshgrid(u, v)
        D2 = U**2 + V**2

        D02 = cutoff**2
        if type == 'low':
            H = np.exp(-D2 / (2 * D02))
        elif type == 'high':
            H = 1 - np.exp(-D2 / (2 * D02))
        else:
            raise ValueError("type must be 'low' or 'high'")

        return H.astype(np.float32)

    @staticmethod
    def filter(img: np.ndarray, cutoff: float, type: str = 'low', verbose: bool = False) -> np.ndarray:
        if len(img.shape) != 2:
            raise ValueError("Chỉ hỗ trợ ảnh xám 2D.")
        
        H, W = img.shape
        t0 = time.perf_counter()

        t1 = time.perf_counter()
        f = np.fft.fft2(img.astype(np.float64))

        t2 = time.perf_counter()
        fshift = np.fft.fftshift(f)


        t3 = time.perf_counter()
        filter_mask = GaussianFreq.create_filter(H, W, cutoff, type)
        
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