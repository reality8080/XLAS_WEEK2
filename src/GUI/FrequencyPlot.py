# gui/frequency_plot.py
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from PyQt5.QtWidgets import QComboBox
from PyQt5.QtCore import Qt, pyqtSlot

class FrequencyPlot(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(4, 3), dpi=100, facecolor='black')
        super().__init__(self.fig)

        # Biến trạng thái
        self.current_mode = '2D'  # '2D', '3D', 'CROSS_SECTION'
        self.freq_data = None

        # Combo box chọn chế độ
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["2D View", "3D Surface", "Vertical Cross-Section"])
        self.mode_combo.setCurrentText("2D View")
        self.mode_combo.currentTextChanged.connect(self.switch_mode)

        # Ban đầu tạo trục 2D
        self.ax = self.fig.add_subplot(111)
        self._setup_2d_style()

    def get_mode_combo(self):
        return self.mode_combo

    def _setup_2d_style(self):
        self.ax.clear()
        self.ax.set_facecolor('black')
        self.fig.patch.set_facecolor('black')
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')

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

    def switch_mode(self, text):
        if '3D' in text:
            self.current_mode = '3D'
        elif 'Cross-Section' in text:
            self.current_mode = 'CROSS_SECTION'
        else:
            self.current_mode = '2D'
        self.plot_frequency(self.freq_data)  # Re-plot with new mode

    @pyqtSlot(np.ndarray)
    def compute_magnitude_spectrum(self, gray_img):
        """Tính phổ biên độ DFT - thread-safe."""
        f = np.fft.fft2(gray_img)
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1e-8)  # Tránh log(0)
        return np.clip(magnitude, 0, 255).astype(np.uint8)

    @pyqtSlot(np.ndarray, str)
    def plot_frequency(self, freq_data: np.ndarray = None, title="Frequency Domain"):
        if freq_data is None:
            return
        self.freq_data = freq_data  # Lưu để switch mode

        h, w = freq_data.shape
        cy, cx = h // 2, w // 2

        if self.current_mode == '2D':
            self.fig.clear()
            self.ax = self.fig.add_subplot(111)
            self._setup_2d_style()
            im = self.ax.imshow(freq_data, cmap='turbo', extent=[-cx, cx, -cy, cy])
            self.ax.set_title(title, color='cyan', fontsize=10)
            self.ax.set_xlabel('u', color='white', fontsize=8)
            self.ax.set_ylabel('v', color='white', fontsize=8)
            self.fig.colorbar(im, ax=self.ax, shrink=0.8)

        elif self.current_mode == 'CROSS_SECTION':
            self._plot_cross_section(freq_data, title)

        elif self.current_mode == '3D':
            self.fig.clear()
            self._setup_3d_style()

            step = max(1, min(w, h) // 180)  # Downsample để nhanh
            small = freq_data[::step, ::step]
            hh, ww = small.shape

            x = np.linspace(-cx, cx, ww)
            y = np.linspace(-cy, cy, hh)
            X, Y = np.meshgrid(x, y)

            surf = self.ax.plot_surface(X, Y, small, cmap='turbo', linewidth=0,
                                        antialiased=False, alpha=0.98, shade=True)
            self.ax.view_init(elev=50, azim=45)
            self.ax.set_zlim(0, small.max() * 1.1)
            self.ax.set_title(title, color='cyan', fontsize=10, pad=20)

        self.fig.subplots_adjust(left=0, right=1, top=0.95, bottom=0)
        self.draw()

    def _plot_cross_section(self, freq_data, title="Vertical Cross-Section"):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self._setup_2d_style()

        h, w = freq_data.shape
        cy = h // 2
        cx = w // 2

        cross_section_data = freq_data[cy, :]

        u_axis = np.linspace(-cx, cx, w)

        self.ax.plot(u_axis, cross_section_data, color='lime', linewidth=2)

        self.ax.set_title(title, color='cyan', fontsize=10)
        self.ax.set_xlabel('u (Horizontal Frequency)', color='white', fontsize=8)
        self.ax.set_ylabel('Magnitude (Log-Scale)', color='white', fontsize=8)
        self.ax.tick_params(colors='white')
        self.ax.grid(axis='y', linestyle='--', alpha=0.5, color='gray')
        self.ax.axvline(x=0, color='red', linestyle=':', linewidth=1)

        self.ax.set_xlim(u_axis.min(), u_axis.max())
        self.ax.set_ylim(0, cross_section_data.max() * 1.1)