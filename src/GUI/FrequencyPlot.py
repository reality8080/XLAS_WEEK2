# gui/frequency_plot.py
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from PyQt5.QtWidgets import QComboBox
from PyQt5.QtCore import Qt

class FrequencyPlot(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(4, 3), dpi=100, facecolor='black')
        super().__init__(self.fig)

        # Biến trạng thái
        self.current_mode = '2D'  # '2D' hoặc '3D'
        self.freq_data = None

        # Combo box chọn chế độ (sẽ được parent thêm vào)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["2D View", "3D Surface"])
        self.mode_combo.setCurrentText("2D View")
        self.mode_combo.currentTextChanged.connect(self.switch_mode)

        # Ban đầu tạo trục 2D
        self.ax = self.fig.add_subplot(111)
        self._setup_2d_style()

    # ====================== CÁC HÀM RIÊNG ======================
    def _setup_2d_style(self):
        self.ax.clear()
        self.ax.set_facecolor('black')
        self.fig.patch.set_facecolor('black')

    def _setup_3d_style(self):
        self.ax.clear()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('black')
        self.ax.grid(False)
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('gray')
        self.ax.yaxis.pane.set_edgecolor('gray')
        self.ax.zaxis.pane.set_edgecolor('gray')
        self.ax.tick_params(colors='white', labelsize=7)
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.zaxis.label.set_color('white')

    # ====================== CHUYỂN ĐỔI CHẾ ĐỘ ======================
    def switch_mode(self, text):
        self.current_mode = '3D' if '3D' in text else '2D'
        if self.freq_data is not None:
            self.plot_frequency(self.freq_data)  # vẽ lại

    def get_mode_combo(self):
        """Trả về combo box để parent thêm vào layout"""
        return self.mode_combo

    # ====================== TÍNH DFT ======================
    def compute_magnitude_spectrum(self, image_gray):
        if image_gray is None or image_gray.size == 0:
            return None
        f = np.fft.fft2(image_gray.astype(np.float64))
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        magnitude_log = np.log(1 + magnitude)
        return magnitude_log

    # ====================== VẼ CHUNG ======================
    def plot_frequency(self, freq_data=None, title="Frequency Domain"):
        self.freq_data = freq_data
        if freq_data is None:
            self.ax.clear()
            self.ax.text(0.5, 0.5, 'No data', transform=self.ax.transAxes,
                        ha='center', va='center', color='white', fontsize=12)
            self.draw()
            return

        h, w = freq_data.shape
        cx, cy = w // 2, h // 2

        if self.current_mode == '2D':
            self.fig.clear()
            self.ax = self.fig.add_subplot(111)  # tạo lại trục 2D
            self._setup_2d_style()

            self.ax.imshow(
                freq_data,
                cmap='turbo',           # đẹp hơn jet
                extent=[-cx, cx, -cy, cy],
                origin='lower',
                interpolation='nearest'
            )
            self.ax.set_title(title, color='cyan', fontsize=10)
            self.ax.set_xlabel('u', color='white', fontsize=8)
            self.ax.set_ylabel('v', color='white', fontsize=8)
            self.ax.tick_params(colors='white')

        else:  # 3D
            self.fig.clear()
            self._setup_3d_style()

            # Downsample thông minh
            step = max(1, min(w, h) // 180)
            small = freq_data[::step, ::step]
            hh, ww = small.shape

            x = np.linspace(-cx, cx, ww)
            y = np.linspace(-cy, cy, hh)
            X, Y = np.meshgrid(x, y)

            surf = self.ax.plot_surface(
                X, Y, small,
                cmap='turbo',
                linewidth=0,
                antialiased=False,
                alpha=0.98,
                shade=True
            )
            self.ax.view_init(elev=50, azim=45)
            self.ax.set_zlim(0, small.max() * 1.1)
            self.ax.set_title(title, color='cyan', fontsize=10, pad=20)

        self.fig.subplots_adjust(left=0, right=1, top=0.95, bottom=0)
        self.draw()