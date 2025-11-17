import numpy as np

class Gamma:
    def __init__(self):
        self.image = None

    def auto_gamma(self,img, gamma_range = (0.4, 2.5), steps = 20):
        best_gamma = 1.0
        best_score = 0
        for gamma in np.linspace(*gamma_range, steps):
            transformed = self.gamma_transform(img, gamma)
            score = np.std(transformed)
            if score>best_score:
                best_score = score
                best_gamma = gamma
        return self.gamma_transform(img, best_gamma), best_gamma

    
    def gamma_transform(self,img, gamma = 0.5):
        # Gamma dieu chinh do sang va do tuong phan cua Anh
        # Chuyen anh vè float và dua mien gia tri ve khoang [0,1]
        img_float = img.astype(np.float32)/255

        # Bien doi gia tri pixel theo so mu gamma
        gamma_img = np.power(img_float, gamma)
        return np.uint8(np.clip(gamma_img*255,0,255))
