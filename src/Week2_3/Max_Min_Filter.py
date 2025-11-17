import numpy as np

class Max_Filter:
    def __init__(self):
        self.image = None
    
    def Filter(self, image:np.ndarray, kernel_size,padding = True):

        if len(image.shape) != 2:
            raise ValueError("Max_Filter chỉ hỗ trợ ảnh xám 2D.") 
        if padding:
            pad_h = kernel_size//2
            pad_w = kernel_size//2

            image = np.pad(image, ((pad_h,pad_h), (pad_w,pad_w)), mode = 'constant',constant_values=0)
        
        h_image,w_image = image.shape

        for i in range(h_image - kernel_size + 1):
            for j in range(w_image - kernel_size + 1):
                s_image = image[i:i+kernel_size, j:j+kernel_size]
                image[i + pad_h, j + pad_w] = np.max(s_image)
        if padding:
            image = image[pad_h:h_image - pad_h, pad_w:w_image - pad_w]
        return image.astype(np.uint8)
    
class Min_Filter:
    def __init__(self):
        self.image = None
    
    def Filter(self, image:np.ndarray, kernel_size,padding = True):

        if len(image.shape) != 2:
            raise ValueError("Min_Filter chỉ hỗ trợ ảnh xám 2D.") 
        if padding:
            pad_h = kernel_size//2
            pad_w = kernel_size//2

            image = np.pad(image, ((pad_h,pad_h), (pad_w,pad_w)), mode = 'constant',constant_values=0)
        
        h_image,w_image = image.shape

        for i in range(h_image - kernel_size + 1):
            for j in range(w_image - kernel_size + 1):
                s_image = image[i:i+kernel_size, j:j+kernel_size]
                image[i + pad_h, j + pad_w] = np.min(s_image)
        if padding:
            image = image[pad_h:h_image - pad_h, pad_w:w_image - pad_w]
        return image.astype(np.uint8)

class Mid_Filter:
    def __init__(self):
        self.image = None
    
    def Filter(self, image:np.ndarray, kernel_size,padding = True):

        if len(image.shape) != 2:
            raise ValueError("Mid_Filter chỉ hỗ trợ ảnh xám 2D.") 
        if padding:
            pad_h = kernel_size//2
            pad_w = kernel_size//2

            image = np.pad(image, ((pad_h,pad_h), (pad_w,pad_w)), mode = 'constant',constant_values=0)
        
        h_image,w_image = image.shape

        for i in range(h_image - kernel_size + 1):
            for j in range(w_image - kernel_size + 1):
                s_image = image[i:i+kernel_size, j:j+kernel_size]
                image[i + pad_h, j + pad_w] = (np.max(s_image) + np.min(s_image)) / 2
        if padding:
            image = image[pad_h:h_image - pad_h, pad_w:w_image - pad_w]
        return image.astype(np.uint8)
          

