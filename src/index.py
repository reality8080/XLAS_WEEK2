from Week2_3.Negative import Negative
from Week2_3.Log_Transform import Log_Transform
from Week2_3.Gamma import Gamma
# from Utils.ImageApp import ImageApp
from GUI.ImageApp import ImageApp
from PyQt5.QtWidgets import QApplication
from Week2_3.Piecewise_Linear_Transform import Piecewise_Linear_Transform
from Week2_3.Histogram import Histogram
from Week2_3.Box_Filter import Box_Filter
import sys
from Week2_3.Meadian_Filter import Meadian_Filter
from Week2_3.Max_Min_Filter import Max_Filter, Min_Filter, Mid_Filter
from Week2_3.Gaussian import Gaussian
from Week2_3.UnsharpMasking_HighBoost import UnsharpMasking_HighBoost
from Week2_3.Laplacian import Laplacian
from Week2_3.Gradien_Sobel import Gradien_Sobel

if __name__ == "__main__":
    app = QApplication(sys.argv)
    negative = Negative()
    log = Log_Transform()
    gamma = Gamma()
    piecewise = Piecewise_Linear_Transform()
    histogram = Histogram()
    box_filter_processor = Box_Filter()
    median_filter_processor = Meadian_Filter()
    max_filter_processor = Max_Filter()
    min_filter_processor = Min_Filter()
    mid_filter_processor = Mid_Filter()
    gaussian_filter_processor = Gaussian()
    unsharp = UnsharpMasking_HighBoost()
    laplacian = Laplacian()
    sobel = Gradien_Sobel()
    
    window = ImageApp(
        negative_processor=negative, 
        log_processor=log, 
        gamma_processor=gamma, 
        piecewise_processor=piecewise,
        histogram_processor=histogram,
        box_filter_processor=box_filter_processor,
        median_filter_processor=median_filter_processor,
        max_filter_processor=max_filter_processor,
        min_filter_processor=min_filter_processor,
        mid_filter_processor=mid_filter_processor,
        gaussian_filter_processor=gaussian_filter_processor,
        unsharp_processor=unsharp,
        laplacian_processor=laplacian,
        sobel_processor=sobel
    )
    window.show()
    sys.exit(app.exec_())
