# gui/frequency_plot.py
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class FrequencyPlot(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(4, 1.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

        # Tùy chỉnh nền và màu chữ cho đẹp
        self.ax.set_facecolor('black')
        self.fig.patch.set_facecolor('white')

    def clear_plot(self):
        self.ax.clear()
        self.ax.set_facecolor('black')
        self.draw()

    def compute_magnitude_spectrum(self, image_gray):
        """Tính phổ biên độ DFT 2D (log + shift)"""
        if image_gray is None or image_gray.size == 0:
            return None
        f = np.fft.fft2(image_gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        # Log scale để thấy rõ tần số thấp
        magnitude_log = np.log(1 + magnitude)
        return magnitude_log

    def plot_frequency(self, freq_data, title="DFT Magnitude", cmap='gray'):
        """
        Vẽ phổ tần số với:
        - Tâm tại (0,0)
        - Tỷ lệ chính xác (không méo)
        - Dễ nhìn (có màu)
        """
        self.ax.clear()
        self.ax.set_facecolor('black')

        if freq_data is None:
            self.ax.text(0.5, 0.5, 'No data', transform=self.ax.transAxes,
                        ha='center', va='center', fontsize=10, color='white')
        else:
            h, w = freq_data.shape
            cx, cy = w // 2, h // 2

            im = self.ax.imshow(
                freq_data,
                cmap=cmap,                    # 'jet', 'hot', 'viridis', 'magma'
                extent=[-cx, cx, -cy, cy],    # Tâm tại (0,0), đúng tỷ lệ
                origin='lower',               # Quan trọng: tần số 0 ở giữa
                interpolation='nearest'
            )

            # Tùy chọn: thêm colorbar nhỏ
            # self.fig.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04, shrink=0.8)

        # Cấu hình trục
        self.ax.set_title(title, fontsize=9, color='white', pad=10)
        self.ax.set_xlabel('Frequency u', fontsize=8, color='white')
        self.ax.set_ylabel('Frequency v', fontsize=8, color='white')
        self.ax.tick_params(colors='white', labelsize=7)

        # Viền trắng cho đẹp
        for spine in self.ax.spines.values():
            spine.set_color('white')

        # Thay tight_layout() bằng subplots_adjust để tránh lỗi Qt
        self.fig.subplots_adjust(left=0.12, right=0.88, top=0.88, bottom=0.18)
        self.draw()