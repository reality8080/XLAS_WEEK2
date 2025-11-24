# Week2_3/FrequencyFilters/GaussianFreq.py
import numpy as np
import time

class GaussianFreq:
    @staticmethod
    def create_filter(H: int, W: int, cutoff: float, type: str = 'low') -> np.ndarray:
        """Tạo mask Gaussian filter (vectorized)."""
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
    def filter(img: np.ndarray, cutoff: float, type: str = 'low', verbose: bool = False) -> tuple:
        """Áp dụng Gaussian frequency filter; hỗ trợ xám/màu; return (result, steps)."""
        if len(img.shape) == 3:  # Hỗ trợ ảnh màu
            channels = [GaussianFreq.filter(img[:, :, c], cutoff, type, verbose=False) for c in range(img.shape[2])]
            result = np.stack([ch[0] for ch in channels], axis=2).astype(np.uint8)
            return result, channels[0][1]
        
        if len(img.shape) != 2:
            raise ValueError("Chỉ hỗ trợ ảnh xám 2D.")
        
        H, W = img.shape
        t0 = time.perf_counter() if verbose else 0

        t1 = time.perf_counter() if verbose else 0
        f = np.fft.fft2(img.astype(np.float64))
        fshift = np.fft.fftshift(f)
        t2 = time.perf_counter() if verbose else 0

        filter_mask = GaussianFreq.create_filter(H, W, cutoff, type)
        t3 = time.perf_counter() if verbose else 0

        fshift_filtered = fshift * filter_mask
        t4 = time.perf_counter() if verbose else 0

        f_ishift = np.fft.ifftshift(fshift_filtered)
        t5 = time.perf_counter() if verbose else 0

        img_back = np.fft.ifft2(f_ishift)
        img_back = np.real(img_back)  # Lấy phần thực
        t6 = time.perf_counter() if verbose else 0

        result = np.clip(img_back, 0, 255).astype(np.uint8)

        steps = {
            "1. FFT2": t2 - t1,
            "2. FFTShift": t3 - t2,
            "3. Tạo bộ lọc": t3 - t2,
            "4. Nhân bộ lọc": t4 - t3,
            "5. IFFTShift": t5 - t4,
            "6. IFFT2": t6 - t5,
            "Tổng thời gian": t6 - t0
        }
        
        if verbose:
            for step, duration in steps.items():
                print(f"{step}: {duration:.4f}s")

        return result, steps