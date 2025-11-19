import numpy as np
from Week2_3.Calculator import Calculator
import cv2

class Laplacian:
    def __init__(self):
        self.image = None

    def Filter(self, image: np.ndarray, kernel_size=3, padding=True, neighborhood=4, scale=0.3):
        """
        Áp dụng Laplacian sharpening.
        - image: Ảnh xám (H, W) uint8.
        - kernel_size: Kích thước kernel lẻ (mặc định 3).
        - padding: Có pad zero không (mặc định True).
        - neighborhood: Số neighbors (4/8/16/full).
        - scale: Hệ số sharpen (0.1-0.5, mặc định 0.3 để tránh over-sharpen).
        """
        if len(image.shape) != 2:
            raise ValueError("Laplacian chỉ hỗ trợ ảnh xám 2D.")
        
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("Kích thước kernel phải là số lẻ và >= 1.")
        
        if neighborhood > kernel_size * kernel_size - 1:
            raise ValueError("Số điểm lân cận không hợp lệ.")
        
        orig_h, orig_w = image.shape  # ← LƯU KÍCH THƯỚC GỐC TRƯỚC PAD
        image = image.astype(np.float32)
        
        pad_size = kernel_size // 2
        B = None  # Để crop sau
        
        if padding:
            # Pad THỦ CÔNG một lần (thêm pad_size mỗi bên)
            padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant', constant_values=0)
            
            # Xây kernel
            kernel_matrix = np.zeros((kernel_size, kernel_size), dtype=np.float32)
            center = kernel_size // 2
            coords = []
            
            if neighborhood == 4:
                coords = [
                    (center - 1, center), (center + 1, center),
                    (center, center - 1), (center, center + 1)
                ]
            elif neighborhood == 8:
                coords = [
                    (center - 1, center), (center + 1, center),
                    (center, center - 1), (center, center + 1),
                    (center - 1, center - 1), (center - 1, center + 1),
                    (center + 1, center - 1), (center + 1, center + 1)
                ]
            elif neighborhood == 16:
                for i in range(kernel_size):
                    for j in range(kernel_size):
                        if abs(i - center) + abs(j - center) <= kernel_size // 2 and (i, j) != (center, center):
                            coords.append((i, j))
            else:
                # Full neighbors trong kernel
                coords = [(i, j) for i in range(kernel_size) for j in range(kernel_size) if (i, j) != (center, center)]
                coords = coords[:neighborhood]  # Nếu vượt, lấy subset
            
            for (i, j) in coords:
                kernel_matrix[i, j] = -1
            kernel_matrix[center, center] = len(coords)
            
            # Convolution trên padded_image, KHÔNG pad thêm (padding=False để tránh double)
            B = Calculator.convolution(padded_image, kernel_matrix, padding=False)
            
            # Crop B về kích thước gốc (loại bỏ pad)
            B = B[pad_size : -pad_size, pad_size : -pad_size]
        else:
            # Không pad: Convolution trực tiếp, output có thể nhỏ hơn (valid mode)
            # Xây kernel tương tự...
            kernel_matrix = np.zeros((kernel_size, kernel_size), dtype=np.float32)
            # ... (code xây coords và kernel giống trên)
            for (i, j) in coords:
                kernel_matrix[i, j] = -1
            kernel_matrix[center, center] = len(coords)
            
            B = Calculator.convolution(image, kernel_matrix, padding=True)  # Dùng padding=True của conv để giữ size
        
        # Công thức sharpen: original + scale * B (B ≈ -Laplacian → +B brighten local max)
        result = image + scale * B
        
        # Đảm bảo shape khớp original và clip
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        self.image = result
        return self.image