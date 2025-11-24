import numpy as np

class Negative:
    @staticmethod
    def negative(img: np.ndarray) -> np.ndarray:
        """Negative transform."""
        return 255 - img