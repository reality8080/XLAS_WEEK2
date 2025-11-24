# gui/histogram_plot.py
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import pyqtSlot   # THÊM DÒNG NÀY
import numpy as np

class HistogramPlot(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(4, 2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

    def clear_plot(self):
        self.ax.clear()
        self.draw()

    @pyqtSlot(np.ndarray, str)   # BẮT BUỘC PHẢI CÓ DÒNG NÀY
    def plot_histogram(self, hist_norm, title="Histogram"):
        self.ax.clear()
        self.ax.bar(range(256), hist_norm.flatten(), width=1.0, color='cyan', edgecolor='cyan')
        self.ax.set_title(title, fontsize=10, color='white')
        self.ax.set_xlim(0, 255)
        self.ax.set_ylim(0, hist_norm.max() * 1.1)
        self.ax.tick_params(axis='both', which='major', labelsize=8, colors='white')
        self.ax.set_facecolor('black')
        self.fig.patch.set_facecolor('black')
        self.fig.tight_layout()
        self.draw()