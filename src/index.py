from Class.Negative import Negative
from Class.Log_Transform import Log_Transform
from Class.Gamma import Gamma
# from Utils.ImageApp import ImageApp
from GUI.ImageApp import ImageApp
from PyQt5.QtWidgets import QApplication
from Class.Piecewise_Linear_Transform import Piecewise_Linear_Transform
from Class.Histogram import Histogram
from Class.Box_Filter import Box_Filter
import sys
from Class.Meadian_Filter import Meadian_Filter
from Class.Max_Min_Filter import Max_Filter, Min_Filter, Mid_Filter
from Class.Gaussian import Gaussian

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
        gaussian_filter_processor=gaussian_filter_processor
    )
    window.show()
    sys.exit(app.exec_())
