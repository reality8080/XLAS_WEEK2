# src/index.py
from Week2_3.Negative import Negative
from Week2_3.Log_Transform import Log_Transform
from Week2_3.Gamma import Gamma
from GUI.ImageApp import ImageApp
from PyQt5.QtWidgets import QApplication
from Week2_3.Piecewise_Linear_Transform import Piecewise_Linear_Transform
from Week2_3.Histogram import Histogram
from Week2_3.Box_Filter import Box_Filter
import sys
from Week2_3.Median_Filter import Median_Filter  # Sửa: Meadian → Median
from Week2_3.Max_Min_Filter import Max_Filter, Min_Filter, Mid_Filter
from Week2_3.Gaussian import Gaussian
from Week2_3.UnsharpMasking_HighBoost import UnsharpMasking_HighBoost
from Week2_3.Laplacian import Laplacian
from Week2_3.Gradien_Sobel import Gradien_Sobel  # Gợi ý: Đổi tên thành Gradient_Sobel
from Week3.Ideal import Ideal
from Week3.Butterworth import Butterworth
from Week3.GaussianFreq import GaussianFreq

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Không cần instantiate processors: ImageApp dùng static calls nội bộ
    window = ImageApp()  # Init không params, tối ưu memory
    window.show()
    sys.exit(app.exec_())