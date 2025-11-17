import numpy as np

class Calculator:
    def __init__(self):
        pass
    @staticmethod
    def convolution(A:np.ndarray,kernel:np.ndarray, padding = False):
        A=A.astype(np.float32)
        kernel = np.flipud(np.fliplr(kernel))

        hA,wA = A.shape
        hk,wk = kernel.shape

        if hk > hA or wk > wA:
            raise ValueError("Kernel size must be smaller than input matrix.")

        if padding:
            pad_h = hk//2
            pad_w = wk//2

            A = np.pad(A, ((pad_h,pad_h), (pad_w,pad_w)), mode = 'constant',constant_values=0)
            hA,wA = A.shape

        B = np.zeros((hA-hk+1,wA-wk+1),dtype=np.float32)

        for i in range(0,hA-hk+1):
            for j in range(0,wA-wk+1):
                s_A = A[i:i+hk, j:j+wk]
                B[i,j] = np.sum(kernel*s_A)
        
        return np.clip(B, 0, 255).astype(np.uint8)
    
    @staticmethod
    def float_convolution(image:np.ndarray, kernel:np.ndarray, padding = True)->np.ndarray:
        image=image.astype(np.float64)
        kernel = np.flipud(np.fliplr(kernel))

        h_image, w_image = image.shape
        hk,wk = kernel.shape

        if hk > h_image or wk > w_image:
            raise ValueError("Kernel size must be smaller than input matrix.")

        if padding:
            pad_h = hk//2
            pad_w = wk//2

            image = np.pad(image, ((pad_h,pad_h), (pad_w,pad_w)), mode = 'constant',constant_values=0)
        h_image, w_image = image.shape



        B = np.zeros((h_image-hk+1,w_image-wk+1),dtype=np.float64)

        for i in range(0,h_image-hk+1):
            for j in range(0,w_image-wk+1):
                s_A = image[i:i+hk, j:j+wk]
                B[i,j] = np.sum(kernel*s_A)
        
        return B.astype(np.float64)
    