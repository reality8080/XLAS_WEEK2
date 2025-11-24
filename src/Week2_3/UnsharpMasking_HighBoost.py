import numpy as np
import cv2

class UnsharpMasking_HighBoost:
    @staticmethod
    def Filter(image: np.ndarray, kernel_size=3, sigma=1.0, amount=1.0, threshold=0):
        """Unsharp masking với threshold soft."""
        if len(image.shape) != 2:
            raise ValueError("Unsharp Masking chỉ hỗ trợ ảnh xám 2D.")
        if kernel_size % 2 == 0 or kernel_size < 1 or sigma <= 0 or amount < 0:
            raise ValueError("Params invalid.")
        amount = min(amount, 2.0)
        image_f = image.astype(np.float32)
        blurred = cv2.GaussianBlur(image_f, (kernel_size, kernel_size), sigma)
        detail = image_f - blurred
        # **Soft threshold: Giảm artifact**
        if threshold > 0:
            detail = detail * (np.abs(detail) > threshold)
        sharpened = image_f + amount * detail
        return np.clip(sharpened, 0, 255).astype(np.uint8)