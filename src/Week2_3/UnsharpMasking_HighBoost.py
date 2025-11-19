import numpy as np
import cv2

class UnsharpMasking_HighBoost:
    def __init__(self):
        self.image = None
        self.image_blur = None

    def Filter(self, image:np.ndarray, kernel_size=3, sigma=1.0, amount=1.0, threshold=0):
        if len(image.shape) != 2:
            raise ValueError("Unsharp Masking chỉ hỗ trợ ảnh xám 2D.")
        
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("Kích thước kernel phải là số lẻ và >= 1.")
        
        if sigma <= 0:
            raise ValueError("Sigma phải là số dương.")
        
        if amount < 0:
            raise ValueError("Amount phải là số không âm.")
        
        amount = min(amount, 2.0)

        image = image.astype(np.float32)

        # Tạo ảnh mờ bằng Gaussian Blur
        self.image_blur = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

        # Tính phần chi tiết
        detail = image - self.image_blur
        detail = np.clip(detail, -50, 50)
        # Áp dụng ngưỡng nếu cần
        if threshold > 0:
            detail = np.where(np.abs(detail) > threshold, detail, 0)

        # Tạo ảnh sắc nét
        sharpened = image + amount * detail

        # Giới hạn giá trị pixel trong khoảng [0, 255]
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        self.image = sharpened
        return self.image