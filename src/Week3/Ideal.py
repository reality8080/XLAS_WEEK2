import numpy as np
import time

class Ideal:
    @staticmethod
    def create_filter(H: int, W: int, cutoff: float, filter_type: str) -> np.ndarray:
        """Tạo mask Ideal filter (vectorized)."""
        u = np.arange(W) - W // 2
        v = np.arange(H) - H // 2
        U, V = np.meshgrid(u, v)
        D = np.sqrt(U**2 + V**2)

        if filter_type == 'low':
            H = (D <= cutoff).astype(np.float32)
        elif filter_type == 'high':
            H = (D > cutoff).astype(np.float32)
        else:
            raise ValueError("filter_type must be 'low' or 'high'")
        return H

    @staticmethod
    def filter(img: np.ndarray, cutoff: float, type: str = 'low', verbose: bool = False) -> tuple:
        """Áp dụng Ideal filter; hỗ trợ xám/màu; return (result, steps)."""
        if len(img.shape) == 3:  # Hỗ trợ ảnh màu
            channels = [Ideal.filter(img[:, :, c], cutoff, type, verbose=False) for c in range(img.shape[2])]
            result = np.stack([ch[0] for ch in channels], axis=2).astype(np.uint8)
            return result, channels[0][1]
        
        if len(img.shape) != 2:
            raise ValueError("Chỉ hỗ trợ ảnh xám 2D.")
        
        img = img.astype(np.float64)
        H, W = img.shape
        t0 = time.perf_counter() if verbose else 0

        t1 = time.perf_counter() if verbose else 0
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        t2 = time.perf_counter() if verbose else 0

        filter_mask = Ideal.create_filter(H, W, cutoff, type)
        t3 = time.perf_counter() if verbose else 0

        G_shifted = fshift * filter_mask
        t4 = time.perf_counter() if verbose else 0

        G = np.fft.ifftshift(G_shifted)
        t5 = time.perf_counter() if verbose else 0

        img_filtered = np.fft.ifft2(G)
        img_filtered = np.real(img_filtered)  # Lấy phần thực
        t6 = time.perf_counter() if verbose else 0

        result = np.clip(img_filtered, 0, 255).astype(np.uint8)

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