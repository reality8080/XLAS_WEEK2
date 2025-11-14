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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    negative = Negative()
    log = Log_Transform()
    gamma = Gamma()
    piecewise = Piecewise_Linear_Transform()
    histogram = Histogram()
    box_filter_processor = Box_Filter()

    window = ImageApp(
        negative_processor=negative, 
        log_processor=log, 
        gamma_processor=gamma, 
        piecewise_processor=piecewise,
        histogram_processor=histogram,
        box_filter_processor=box_filter_processor
    )
    window.show()
    sys.exit(app.exec_())
