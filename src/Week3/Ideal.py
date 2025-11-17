import numpy as np
from Week2_3.Calculator import Calculator

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
    def filter(img:np.ndarray, cutoff:float, type:str = 'low', padding:bool = True)->np.ndarray:
        if len(img.shape) != 2:
            raise ValueError("Chỉ hỗ trợ ảnh xám 2D.")
        
        img = img.astype(np.float32)
        H, W = img.shape

        # Tính DFT 2D
        f = np.fft.fft2(img)
        f_shifted = np.fft.fftshift(f)

        # Tạo bộ lọc lý tưởng
        filter_mask = Ideal.create_filter(H, W, cutoff, type)

        # Áp dụng bộ lọc
        G_shifted = f_shifted * filter_mask

        # DFT ngược để lấy ảnh đã lọc
        G = np.fft.ifftshift(G_shifted)
        img_filtered = np.fft.ifft2(G)
        img_filtered = np.abs(img_filtered)

        return np.clip(img_filtered, 0, 255).astype(np.uint8)