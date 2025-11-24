import numpy as np
from scipy.ndimage import convolve

class Calculator:
    def __init__(self):
        pass
    @staticmethod
    def convolution(A:np.ndarray,kernel:np.ndarray, padding = False, mode = 'constant') -> np.ndarray:
        A=A.astype(np.float32)
        kernel = np.flipud(np.fliplr(kernel))

        if kernel.shape[0] > A.shape[0] or kernel.shape[1] > A.shape[1]:
            raise ValueError("Kernel size must be smaller than input matrix.")

        if padding:
            result = convolve(A, kernel, mode=mode, cval=0.0)
        else:
            result = convolve(A, kernel, mode='valid', cval=0.0)

        if A.dtype == np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)
        return result.astype(np.float32)