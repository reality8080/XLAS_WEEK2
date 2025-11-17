import cv2
import numpy as np
from Class.Calculator import Calculator
class Gradien_Sobel:
    def __init__(self):
        pass

    def binomial_coeffs(self,n):
        coeffs = [1]
        for k in range(1, n+1):
            coeffs.append(coeffs[-1]*(n-k+1)//k)
        return np.array(coeffs, dtype=np.float64)

    def get_sobel_kernels(self, ksize, dx,dy):
        if ksize%2==0 or ksize<3:
            raise ValueError("Kernel size must be an odd integer greater than or equal to 3.")
        
        s_coeffs = self.binomial_coeffs(ksize-1)
        s_kernel = s_coeffs.reshape(-1, 1)
        d_coeffs = np.arange(-(ksize//2), ksize//2+1)
        d_kernel = d_coeffs*s_coeffs

        if dx == 1 and dy == 0:
            kernel_2d = s_kernel@d_kernel.reshape(1, -1)
        elif dx == 0 and dy == 1:
            kernel_2d = d_kernel.reshape(-1, 1)@s_coeffs.reshape(1, -1)
        else:
            raise ValueError("Only first order derivatives in x or y direction are supported.")
        
        return kernel_2d.astype(np.float64)
    

    def apply_sobel(self,image: np.ndarray, ddepth: int, dx: int, dy: int, ksize=3, scale=1.0, delta=0.0):

        if not (dx == 1 and dy == 0) and not (dx == 0 and dy == 1):
            raise ValueError("Sobel chỉ hỗ trợ (dx=1, dy=0) cho đạo hàm X hoặc (dx=0, dy=1) cho đạo hàm Y.")
        try:
            kernel = self.get_sobel_kernels(ksize, dx, dy)
        except ValueError as e:
            raise e

            # Convert to grayscale if the image is in color
        dst = Calculator.float_convolution(image, kernel, padding=True)

        dst = dst * scale + delta
        if ddepth == cv2.CV_8U:
            abs_dst = np.absolute(dst)
            final_dst = np.clip(abs_dst, 0, 255).astype(np.uint8)
        elif ddepth == cv2.CV_16U:
            final_dst = np.clip(dst, 0, 65535).astype(np.uint16)
        elif ddepth == cv2.CV_16S:
            final_dst = np.clip(dst, -32768, 32767).astype(np.int16)
        elif ddepth == cv2.CV_32F:
            final_dst = dst.astype(np.float32)
        elif ddepth == cv2.CV_64F:
            final_dst = dst.astype(np.float64)
        else:
            final_dst = dst.astype(np.float64)
        return final_dst