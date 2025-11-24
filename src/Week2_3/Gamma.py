import numpy as np

class Gamma:
    def __init__(self):
        self.image = None

    def auto_gamma(self, img, gamma_range=(0.4, 2.5), steps=20):
        """Tìm gamma tối ưu dựa trên entropy (tốt hơn std)."""
        best_gamma = 1.0
        best_score = -np.inf  # **Dùng entropy để đo contrast**
        gammas = np.linspace(*gamma_range, steps)
        
        img_flat = img.flatten() / 255.0  # **Vectorize flatten một lần**
        for gamma in gammas:
            transformed = np.power(img_flat, gamma) * 255
            hist, _ = np.histogram(transformed, bins=256, range=(0, 255), density=True)
            hist = hist[hist > 0]  # Tránh log(0)
            score = -np.sum(hist * np.log2(hist))  # Entropy
            if score > best_score:
                best_score = score
                best_gamma = gamma
        return self.gamma_transform(img, best_gamma), best_gamma

    
    def gamma_transform(self,img:np.ndarray, gamma = 0.5):
        # Gamma dieu chinh do sang va do tuong phan cua Anh
        # Chuyen anh vè float và dua mien gia tri ve khoang [0,1]
        img_float = img.astype(np.float32)/255

        # Bien doi gia tri pixel theo so mu gamma
        gamma_img = np.power(img_float, gamma)*255
        return np.uint8(np.clip(gamma_img,0,255))
