import numpy as np
import time

class Ideal:
    @staticmethod
    def create_filter(H:int,W:int,cutoff:float,filter_type:str)->np.ndarray:
        u = np.arange(W)-W//2
        v = np.arange(H)-H//2
        U, V = np.meshgrid(u,v)
        D = np.sqrt(U**2 + V**2)

        if filter_type == 'low':
            H = (D <= cutoff).astype(np.float32)
        elif filter_type == 'high':
            H = (D > cutoff).astype(np.float32)
        else:
            raise ValueError("filter_type must be 'low' or 'high'")
        return H
    

    @staticmethod
    def filter(img: np.ndarray, cutoff: float, type: str = 'low', padding: bool = True, verbose: bool = False) -> np.ndarray:
        if len(img.shape) != 2:
            raise ValueError("Chỉ hỗ trợ ảnh xám 2D.")
        
        img = img.astype(np.float32)
        H, W = img.shape

        t0 = time.perf_counter()

        t1 = time.perf_counter()
        f = np.fft.fft2(img)
        t2 = time.perf_counter()

        f_shifted = np.fft.fftshift(f)
        t3 = time.perf_counter()

        filter_mask = Ideal.create_filter(H, W, cutoff, type)
        t4 = time.perf_counter()

        G_shifted = f_shifted * filter_mask
        t5 = time.perf_counter()

        G = np.fft.ifftshift(G_shifted)
        t6 = time.perf_counter()

        img_filtered = np.fft.ifft2(G)
        img_filtered = np.abs(img_filtered)
        t7 = time.perf_counter()

        result = np.clip(img_filtered, 0, 255).astype(np.uint8)

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