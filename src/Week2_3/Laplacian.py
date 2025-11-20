import numpy as np
import cv2

class Laplacian:
    def __init__(self):
        self.image = None

    def _get_laplacian_kernel(self, kernel_size: int, neighborhood: int) -> np.ndarray:
        """
        Tạo kernel Laplacian chuẩn dựa trên kích thước và số lân cận.
        Chỉ hỗ trợ kernel_size = 3 (cho 4-neighbors và 8-neighbors chuẩn).
        """
        # Nếu không phải kernel 3x3, ta sẽ dùng kernel đơn giản 8-neighbors
        if kernel_size != 3:
            # Dùng kernel 3x3 chuẩn làm mặc định cho kernel lớn hơn
            # Hoặc trả về lỗi tùy vào yêu cầu của hệ thống
            print(f"Cảnh báo: Kernel size {kernel_size} không chuẩn. Dùng kernel 3x3 (8-neighbors) thay thế.")
            kernel_size = 3
            neighborhood = 8 

        if kernel_size == 3:
            if neighborhood == 4:
                # 4-neighbors kernel:
                # [[ 0,  1,  0],
                #  [ 1, -4,  1],
                #  [ 0,  1,  0]]
                kernel = np.array([
                    [ 0,  1,  0],
                    [ 1, -4,  1],
                    [ 0,  1,  0]
                ], dtype=np.float32)
            elif neighborhood >= 8: # Mặc định cho 8, 16, full (ta chỉ dùng kernel 3x3 nên 8 là tối đa)
                # 8-neighbors kernel (phổ biến nhất):
                # [[ 1,  1,  1],
                #  [ 1, -8,  1],
                #  [ 1,  1,  1]]
                kernel = np.array([
                    [ 1,  1,  1],
                    [ 1, -8,  1],
                    [ 1,  1,  1]
                ], dtype=np.float32)
            else:
                 # 4-neighbors là mặc định
                 kernel = np.array([
                    [ 0,  1,  0],
                    [ 1, -4,  1],
                    [ 0,  1,  0]
                ], dtype=np.float32)
        else:
            # Dùng Laplacian của OpenCV (kernel=1 tương đương 3x3, 8-neighbors)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            kernel = np.float32(kernel)
            kernel[1, 1] = 1 - np.sum(kernel) # Tính lại tâm: 1 - 8 = -7. Ta sẽ dùng -8
            kernel[1, 1] = -8 # Dùng -8 theo chuẩn 8-neighbors
            
        return kernel


    def Filter(self, image: np.ndarray, kernel_size=3, neighborhood=4, scale=0.3):
        """
        Áp dụng Laplacian sharpening bằng cv2.filter2D.
        - image: Ảnh xám (H, W) uint8.
        - kernel_size: Kích thước kernel lẻ (mặc định 3, chỉ hỗ trợ chuẩn 3x3).
        - neighborhood: Số neighbors (4 hoặc >= 8).
        - scale: Hệ số sharpen (0.1-0.5, mặc định 0.3).
        """
        if len(image.shape) != 2:
            raise ValueError("Laplacian chỉ hỗ trợ ảnh xám 2D.")
        
        # 1. Chuẩn bị ảnh và kernel
        image_float = image.astype(np.float32)
        
        # Lấy kernel
        kernel_matrix = self._get_laplacian_kernel(kernel_size, neighborhood)
        
        # 2. Tính Laplacian (B)
        # BORDER_REPLICATE giúp xử lý tốt hơn ở biên so với BORDER_CONSTANT (zero pad)
        # Sử dụng cv2.filter2D cho tốc độ cao nhất (thực hiện convolution)
        B = cv2.filter2D(image_float, -1, kernel_matrix, borderType=cv2.BORDER_REPLICATE)
        
        # 3. Công thức Sharpening: G_sharpen = G_original - B
        # Lưu ý: Kernel Laplacian trong OpenCV thường được định nghĩa với tâm dương (ví dụ: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]) 
        # → B (kết quả convolution) là *tính gần đúng* đạo hàm bậc hai.
        # Sharpening: G' = G - B (nếu B là kernel dương, tâm âm: [[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        # Hoặc: G' = G + B (nếu B là kernel âm, tâm dương: [[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        # Kernel của ta (4-neighbors): [[0, 1, 0], [1, -4, 1], [0, 1, 0]] → Tâm âm. Dùng công thức G - B
        
        # Laplacian Sharpened = Original + scale * B
        # Do kernel của ta có tâm âm, B sẽ mang dấu dương (nếu pixel sáng hơn lân cận).
        # Ta dùng công thức Original + scale * B
        
        result = image_float - scale * B  # Dùng trừ vì kernel có tâm âm (như kernel chuẩn trong sách Gonzalez)
        
        # 4. Clip và Cast
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        self.image = result
        return self.image