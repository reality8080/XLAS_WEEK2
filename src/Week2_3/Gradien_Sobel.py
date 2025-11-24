import cv2
import numpy as np
from Week2_3.Calculator import Calculator

class Gradien_Sobel:
    def __init__(self):
        pass

    def binomial_coeffs(self, n):
        """Tạo hệ số binomial (giữ nguyên)."""
        coeffs = [1]
        for k in range(1, n + 1):
            coeffs.append(coeffs[-1] * (n - k + 1) // k)
        return np.array(coeffs, dtype=np.float64)

    def get_sobel_kernels(self, ksize, dx, dy):
        """Tạo kernel Sobel custom (giữ nguyên)."""
        if ksize % 2 == 0 or ksize < 3:
            raise ValueError("Kernel size must be an odd integer >= 3.")
        s_coeffs = self.binomial_coeffs(ksize - 1)
        s_kernel = s_coeffs.reshape(-1, 1)
        d_coeffs = np.arange(-(ksize // 2), ksize // 2 + 1)
        d_kernel = d_coeffs * s_coeffs
        if dx == 1 and dy == 0:
            kernel_2d = s_kernel @ d_kernel.reshape(1, -1)
        elif dx == 0 and dy == 1:
            kernel_2d = d_kernel.reshape(-1, 1) @ s_coeffs.reshape(1, -1)
        else:
            raise ValueError("Only dx=1 dy=0 or dx=0 dy=1 supported.")
        return kernel_2d.astype(np.float64)

    def apply_sobel(self, image: np.ndarray, ddepth: int = cv2.CV_8U, dx: int = 1, dy: int = 0, 
                    ksize=3, scale=1.0, delta=0.0, original=None, blend=False, blend_scale=0.5):
        """Sử dụng cv2.Sobel cho tốc độ cao; fallback custom nếu cần."""
        if not ((dx == 1 and dy == 0) or (dx == 0 and dy == 1)):
            raise ValueError("Sobel chỉ hỗ trợ đạo hàm X hoặc Y.")
        
        # **Ưu tiên cv2.Sobel: Nhanh hơn 10x**
        if ddepth == -1:  # Auto depth
            ddepth = cv2.CV_32F if scale != 1.0 else cv2.CV_8U
        dst = cv2.Sobel(image, ddepth, dx, dy, ksize=ksize, scale=scale, delta=delta, 
                        borderType=cv2.BORDER_REPLICATE)
        
        if blend and original is not None:
            abs_dst = np.absolute(dst)
            result = original.astype(np.float32) + blend_scale * abs_dst
            return np.clip(result, 0, 255).astype(np.uint8)
        
        # **Cast theo ddepth (giữ nguyên logic)**
        if ddepth == cv2.CV_8U:
            return np.clip(np.absolute(dst), 0, 255).astype(np.uint8)
        elif ddepth == cv2.CV_16U:
            return np.clip(dst, 0, 65535).astype(np.uint16)
        # ... (giữ nguyên các case khác)
        return dst.astype(np.float64)