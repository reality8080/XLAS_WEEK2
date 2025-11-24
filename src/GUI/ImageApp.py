# gui/image_app_gui.py
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog,
    QHBoxLayout, QMessageBox, QGroupBox, QFormLayout, QComboBox,
    QTextEdit, QSpinBox, QDoubleSpinBox, QCheckBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QMetaObject, Q_ARG
import cv2
import numpy as np

# Import HistogramPlot và FrequencyPlot
from .histogram_plot import HistogramPlot 
from .FrequencyPlot import FrequencyPlot

# Import các processor từ Week2_3 (sửa lỗi chính tả và thống nhất static calls)
from Week2_3.Negative import Negative
from Week2_3.Histogram import Histogram
from Week2_3.Log_Transform import Log_Transform
from Week2_3.Gamma import Gamma
from Week2_3.Piecewise_Linear_Transform import Piecewise_Linear_Transform
from Week2_3.Box_Filter import Box_Filter
from Week2_3.Median_Filter import Median_Filter  # Sửa: Meadian → Median
from Week2_3.Max_Min_Filter import Max_Filter, Min_Filter, Mid_Filter
from Week2_3.Gaussian import Gaussian
from Week2_3.UnsharpMasking_HighBoost import UnsharpMasking_HighBoost
from Week2_3.Laplacian import Laplacian
from Week2_3.Gradien_Sobel import Gradien_Sobel  # Giữ nguyên tên class, nhưng note: Có thể sửa thành Gradient_Sobel nếu file rename

# Import frequency filters từ Week3
from Week3.Ideal import Ideal
from Week3.Butterworth import Butterworth
from Week3.GaussianFreq import GaussianFreq

class ImageProcessorThread(QThread):
    result_ready = pyqtSignal(object, str)  # ĐỔI np.ndarray → object
    # hoặc tốt hơn: dùng object để linh hoạt
    # result_ready = pyqtSignal([np.ndarray, str], [str, str])  # overload (khó)

    def __init__(self, img, processor_func, *args, **kwargs):
        super().__init__()
        self.img = img.copy() if img is not None else None  # Tránh reference lỗi
        self.processor_func = processor_func
        self.args = args
        self.kwargs = kwargs

    @pyqtSlot()
    def run(self):
        try:
            result = self.processor_func(self.img, *self.args, **self.kwargs)
            # Frequency filter trả về tuple (img, steps)
            if isinstance(result, tuple) and len(result) == 2:
                img_out = result[0]
            else:
                img_out = result
            self.result_ready.emit(img_out, self.processor_func.__name__)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.result_ready.emit(self.img, f"ERROR: {str(e)}")  # Không emit None!


class ImageApp(QWidget):
    def __init__(self):
        super().__init__()
        # Không cần instances vì hầu hết là static methods
        self.executor = ThreadPoolExecutor(max_workers=4)  # Tăng workers cho multi-task
        self.original_image = None
        self.transformed_image = None
        self.histogram_processor = Histogram()  # Giữ instance nếu cần state
        
        self.active_threads = []

        self.init_ui()

    def init_ui(self):
        """Cập nhật giao diện: Thêm groups cho spatial và frequency filters, params động"""
        main_layout = QHBoxLayout()

        # Left: Image displays
        images_group = QGroupBox("Images")
        images_layout = QVBoxLayout()
        self.original_label = QLabel("Original Image")
        self.original_label.setFixedSize(400, 400)
        self.transformed_label = QLabel("Transformed Image")
        self.transformed_label.setFixedSize(400, 400)
        images_layout.addWidget(self.original_label)
        images_layout.addWidget(self.transformed_label)
        images_group.setLayout(images_layout)
        main_layout.addWidget(images_group)

        # Middle: Plots
        plots_group = QGroupBox("Plots")
        plots_layout = QVBoxLayout()

        # Histogram plots
        self.original_hist_canvas = HistogramPlot(self)
        self.transformed_hist_canvas = HistogramPlot(self)
        plots_layout.addWidget(QLabel("Original Histogram"))
        plots_layout.addWidget(self.original_hist_canvas)
        plots_layout.addWidget(QLabel("Transformed Histogram"))
        plots_layout.addWidget(self.transformed_hist_canvas)

        # Frequency plots với mode combo
        self.original_freq_canvas = FrequencyPlot(self)
        self.transformed_freq_canvas = FrequencyPlot(self)
        plots_layout.addWidget(QLabel("Original Frequency"))
        plots_layout.addWidget(self.original_freq_canvas.get_mode_combo())  # Thêm combo
        plots_layout.addWidget(self.original_freq_canvas)
        plots_layout.addWidget(QLabel("Transformed Frequency"))
        plots_layout.addWidget(self.transformed_freq_canvas.get_mode_combo())  # Thêm combo
        plots_layout.addWidget(self.transformed_freq_canvas)

        plots_group.setLayout(plots_layout)
        main_layout.addWidget(plots_group)

        # Right: Controls
        controls_group = QGroupBox("Controls")
        controls_layout = QFormLayout()

        # Load button
        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)
        controls_layout.addRow(load_btn)

        # Spatial Domain Filters Group
        spatial_group = QGroupBox("Spatial Domain Filters")
        spatial_layout = QVBoxLayout()

        # Negative
        negative_btn = QPushButton("Negative")
        negative_btn.clicked.connect(lambda: self.apply_filter(Negative.negative))
        spatial_layout.addWidget(negative_btn)

        # Histogram Equalization
        hist_eq_btn = QPushButton("Histogram Equalization")
        hist_eq_btn.clicked.connect(lambda: self.apply_filter(self.histogram_processor.histogram_Equalization))
        spatial_layout.addWidget(hist_eq_btn)

        # Log Transform
        log_group = QHBoxLayout()
        self.log_c_spin = QDoubleSpinBox(value=1.0, minimum=0.1, maximum=10.0, singleStep=0.1)
        log_btn = QPushButton("Log Transform")
        log_btn.clicked.connect(lambda: self.apply_filter(Log_Transform().log_transform, c=self.log_c_spin.value(), flag=False))
        log_group.addWidget(QLabel("c:"))
        log_group.addWidget(self.log_c_spin)
        log_group.addWidget(log_btn)
        spatial_layout.addLayout(log_group)

        # Gamma Transform
        gamma_group = QHBoxLayout()
        self.gamma_spin = QDoubleSpinBox(value=0.5, minimum=0.1, maximum=5.0, singleStep=0.1)
        gamma_btn = QPushButton("Gamma Transform")
        gamma_btn.clicked.connect(lambda: self.apply_filter(Gamma().gamma_transform, gamma=self.gamma_spin.value()))
        gamma_group.addWidget(QLabel("Gamma:"))
        gamma_group.addWidget(self.gamma_spin)
        gamma_group.addWidget(gamma_btn)
        spatial_layout.addLayout(gamma_group)

        # Auto Gamma
        auto_gamma_btn = QPushButton("Auto Gamma")
        auto_gamma_btn.clicked.connect(lambda: self.apply_filter(Gamma().auto_gamma))
        spatial_layout.addWidget(auto_gamma_btn)

        # Piecewise Linear
        piecewise_group = QHBoxLayout()
        self.r1_spin = QSpinBox(value=70, minimum=0, maximum=255)
        self.r2_spin = QSpinBox(value=140, minimum=0, maximum=255)
        piecewise_btn = QPushButton("Piecewise Linear")
        piecewise_btn.clicked.connect(lambda: self.apply_filter(Piecewise_Linear_Transform().piecewise_linear_transform, r1=self.r1_spin.value(), r2=self.r2_spin.value()))
        piecewise_group.addWidget(QLabel("r1:"))
        piecewise_group.addWidget(self.r1_spin)
        piecewise_group.addWidget(QLabel("r2:"))
        piecewise_group.addWidget(self.r2_spin)
        piecewise_group.addWidget(piecewise_btn)
        spatial_layout.addLayout(piecewise_group)

        # Box Filter
        box_group = QHBoxLayout()
        self.box_ksize_spin = QSpinBox(value=3, minimum=1, maximum=15, singleStep=2)
        box_padding_check = QCheckBox("Padding")
        box_padding_check.setChecked(True)
        box_btn = QPushButton("Box Filter")
        box_btn.clicked.connect(lambda: self.apply_filter(Box_Filter.filter, kernel_size=self.box_ksize_spin.value(), padding=box_padding_check.isChecked()))
        box_group.addWidget(QLabel("ksize:"))
        box_group.addWidget(self.box_ksize_spin)
        box_group.addWidget(box_padding_check)
        box_group.addWidget(box_btn)
        spatial_layout.addLayout(box_group)

        # Median Filter
        median_group = QHBoxLayout()
        self.median_ksize_spin = QSpinBox(value=3, minimum=1, maximum=15, singleStep=2)
        median_padding_check = QCheckBox("Padding")
        median_padding_check.setChecked(True)
        median_btn = QPushButton("Median Filter")
        median_btn.clicked.connect(lambda: self.apply_filter(Median_Filter.Filter, kernel_size=self.median_ksize_spin.value(), padding=median_padding_check.isChecked()))
        median_group.addWidget(QLabel("ksize:"))
        median_group.addWidget(self.median_ksize_spin)
        median_group.addWidget(median_padding_check)
        median_group.addWidget(median_btn)
        spatial_layout.addLayout(median_group)

        # Max/Min/Mid Filters
        minmax_group = QHBoxLayout()
        self.minmax_ksize_spin = QSpinBox(value=3, minimum=1, maximum=15, singleStep=2)
        minmax_type_combo = QComboBox()
        minmax_type_combo.addItems(["Max", "Min", "Mid"])
        minmax_padding_check = QCheckBox("Padding")
        minmax_padding_check.setChecked(True)
        minmax_btn = QPushButton("Apply")
        minmax_btn.clicked.connect(lambda: self.apply_minmax_filter(minmax_type_combo.currentText(), self.minmax_ksize_spin.value(), minmax_padding_check.isChecked()))
        minmax_group.addWidget(QLabel("Type:"))
        minmax_group.addWidget(minmax_type_combo)
        minmax_group.addWidget(QLabel("ksize:"))
        minmax_group.addWidget(self.minmax_ksize_spin)
        minmax_group.addWidget(minmax_padding_check)
        minmax_group.addWidget(minmax_btn)
        spatial_layout.addLayout(minmax_group)

        # Gaussian Filter
        gaussian_group = QHBoxLayout()
        self.gaussian_ksize_spin = QSpinBox(value=3, minimum=1, maximum=15, singleStep=2)
        self.gaussian_sigma_spin = QDoubleSpinBox(value=1.0, minimum=0.1, maximum=10.0, singleStep=0.1)
        gaussian_padding_check = QCheckBox("Padding")
        gaussian_padding_check.setChecked(True)
        gaussian_btn = QPushButton("Gaussian Filter")
        gaussian_btn.clicked.connect(lambda: self.apply_filter(Gaussian.filter, kernel_size=self.gaussian_ksize_spin.value(), sigma=self.gaussian_sigma_spin.value(), padding=gaussian_padding_check.isChecked()))
        gaussian_group.addWidget(QLabel("ksize:"))
        gaussian_group.addWidget(self.gaussian_ksize_spin)
        gaussian_group.addWidget(QLabel("sigma:"))
        gaussian_group.addWidget(self.gaussian_sigma_spin)
        gaussian_group.addWidget(gaussian_padding_check)
        gaussian_group.addWidget(gaussian_btn)
        spatial_layout.addLayout(gaussian_group)

        # Unsharp Masking
        unsharp_group = QHBoxLayout()
        self.unsharp_ksize_spin = QSpinBox(value=3, minimum=1, maximum=15, singleStep=2)
        self.unsharp_sigma_spin = QDoubleSpinBox(value=1.0, minimum=0.1, maximum=10.0, singleStep=0.1)
        self.unsharp_amount_spin = QDoubleSpinBox(value=1.0, minimum=0.0, maximum=2.0, singleStep=0.1)
        self.unsharp_threshold_spin = QSpinBox(value=0, minimum=0, maximum=255)
        unsharp_btn = QPushButton("Unsharp Masking")
        unsharp_btn.clicked.connect(lambda: self.apply_filter(UnsharpMasking_HighBoost.Filter, kernel_size=self.unsharp_ksize_spin.value(), sigma=self.unsharp_sigma_spin.value(), amount=self.unsharp_amount_spin.value(), threshold=self.unsharp_threshold_spin.value()))
        unsharp_group.addWidget(QLabel("ksize:"))
        unsharp_group.addWidget(self.unsharp_ksize_spin)
        unsharp_group.addWidget(QLabel("sigma:"))
        unsharp_group.addWidget(self.unsharp_sigma_spin)
        unsharp_group.addWidget(QLabel("amount:"))
        unsharp_group.addWidget(self.unsharp_amount_spin)
        unsharp_group.addWidget(QLabel("thresh:"))
        unsharp_group.addWidget(self.unsharp_threshold_spin)
        unsharp_group.addWidget(unsharp_btn)
        spatial_layout.addLayout(unsharp_group)

        # Laplacian
        laplacian_group = QHBoxLayout()
        self.laplacian_ksize_spin = QSpinBox(value=3, minimum=1, maximum=15, singleStep=2)
        self.laplacian_scale_spin = QDoubleSpinBox(value=0.3, minimum=0.0, maximum=1.0, singleStep=0.1)
        laplacian_btn = QPushButton("Laplacian Sharpen")
        laplacian_btn.clicked.connect(lambda: self.apply_filter(Laplacian().Filter, ksize=self.laplacian_ksize_spin.value(), scale=self.laplacian_scale_spin.value()))
        laplacian_group.addWidget(QLabel("ksize:"))
        laplacian_group.addWidget(self.laplacian_ksize_spin)
        laplacian_group.addWidget(QLabel("scale:"))
        laplacian_group.addWidget(self.laplacian_scale_spin)
        laplacian_group.addWidget(laplacian_btn)
        spatial_layout.addLayout(laplacian_group)

        # Sobel Gradient
        sobel_group = QHBoxLayout()
        sobel_dx_check = QCheckBox("dx")
        sobel_dx_check.setChecked(True)
        sobel_dy_check = QCheckBox("dy")
        self.sobel_ksize_spin = QSpinBox(value=3, minimum=3, maximum=31, singleStep=2)
        self.sobel_scale_spin = QDoubleSpinBox(value=1.0, minimum=0.1, maximum=10.0, singleStep=0.1)
        self.sobel_blend_check = QCheckBox("Blend with Original")
        self.sobel_blend_scale_spin = QDoubleSpinBox(value=0.5, minimum=0.0, maximum=1.0, singleStep=0.1)
        sobel_btn = QPushButton("Sobel Gradient")
        sobel_btn.clicked.connect(lambda: self.apply_sobel_filter(sobel_dx_check.isChecked(), sobel_dy_check.isChecked(), self.sobel_ksize_spin.value(), self.sobel_scale_spin.value(), self.sobel_blend_check.isChecked(), self.sobel_blend_scale_spin.value()))
        sobel_group.addWidget(sobel_dx_check)
        sobel_group.addWidget(sobel_dy_check)
        sobel_group.addWidget(QLabel("ksize:"))
        sobel_group.addWidget(self.sobel_ksize_spin)
        sobel_group.addWidget(QLabel("scale:"))
        sobel_group.addWidget(self.sobel_scale_spin)
        sobel_group.addWidget(self.sobel_blend_check)
        sobel_group.addWidget(QLabel("blend scale:"))
        sobel_group.addWidget(self.sobel_blend_scale_spin)
        sobel_group.addWidget(sobel_btn)
        spatial_layout.addLayout(sobel_group)

        spatial_group.setLayout(spatial_layout)
        controls_layout.addRow(spatial_group)

        # Frequency Domain Filters Group
        freq_group = QGroupBox("Frequency Domain Filters")
        freq_layout = QVBoxLayout()

        # Common params
        freq_params_group = QHBoxLayout()
        self.freq_cutoff_spin = QDoubleSpinBox(value=30.0, minimum=1.0, maximum=500.0, singleStep=1.0)
        self.freq_order_spin = QSpinBox(value=1, minimum=1, maximum=10)
        self.freq_type_combo = QComboBox()
        self.freq_type_combo.addItems(["low", "high"])
        freq_params_group.addWidget(QLabel("Cutoff:"))
        freq_params_group.addWidget(self.freq_cutoff_spin)
        freq_params_group.addWidget(QLabel("Order:"))
        freq_params_group.addWidget(self.freq_order_spin)
        freq_params_group.addWidget(QLabel("Type:"))
        freq_params_group.addWidget(self.freq_type_combo)
        freq_layout.addLayout(freq_params_group)

        # Ideal
        ideal_btn = QPushButton("Ideal Filter")
        ideal_btn.clicked.connect(lambda: self.apply_filter(Ideal.filter, cutoff=self.freq_cutoff_spin.value(), type=self.freq_type_combo.currentText()))
        freq_layout.addWidget(ideal_btn)

        # Butterworth
        butter_btn = QPushButton("Butterworth Filter")
        butter_btn.clicked.connect(lambda: self.apply_filter(Butterworth.filter, cutoff=self.freq_cutoff_spin.value(), order=self.freq_order_spin.value(), type=self.freq_type_combo.currentText()))
        freq_layout.addWidget(butter_btn)

        # Gaussian Freq
        gauss_freq_btn = QPushButton("Gaussian Freq Filter")
        gauss_freq_btn.clicked.connect(lambda: self.apply_filter(GaussianFreq.filter, cutoff=self.freq_cutoff_spin.value(), type=self.freq_type_combo.currentText()))
        freq_layout.addWidget(gauss_freq_btn)

        freq_group.setLayout(freq_layout)
        controls_layout.addRow(freq_group)

        # Log output for timing/verbose
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        controls_layout.addRow(QLabel("Log:"))
        controls_layout.addRow(self.log_text)

        controls_group.setLayout(controls_layout)
        main_layout.addWidget(controls_group)

        self.setLayout(main_layout)
        self.setWindowTitle("Image Processing App")
        self.show()

    def load_image(self):
        """Load và display original image"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp)")
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)  # RGB cho display
            self.display_image(self.original_image, self.original_label)
            self.executor.submit(self.update_original_histogram, self.original_image)
            self.executor.submit(self.update_original_display, self.original_image)

    def apply_filter(self, func, *args, **kwargs):
        if self.original_image is None:
            return
        
        thread = ImageProcessorThread(self.original_image, func, *args, **kwargs)
        thread.result_ready.connect(self.on_filter_result)
        
        # Giữ reference + tự động xóa khi xong
        thread.finished.connect(lambda: self.active_threads.remove(thread))
        self.active_threads.append(thread)
        
        thread.start()

    def apply_minmax_filter(self, filter_type, ksize, padding):
        """Handler cho Max/Min/Mid"""
        if filter_type == "Max":
            self.apply_filter(Max_Filter.Filter, kernel_size=ksize, padding=padding)
        elif filter_type == "Min":
            self.apply_filter(Min_Filter.Filter, kernel_size=ksize, padding=padding)
        elif filter_type == "Mid":
            self.apply_filter(Mid_Filter.Filter, kernel_size=ksize, padding=padding)

    def apply_sobel_filter(self, dx, dy, ksize, scale, blend, blend_scale):
        if dx and dy:
            QMessageBox.warning(self, "Lỗi", "Chỉ chọn 1 trong 2: dx hoặc dy!")
            return
        if not dx and not dy:
            QMessageBox.warning(self, "Lỗi", "Phải chọn ít nhất dx hoặc dy!")
            return
        
        sobel_dx = 1 if dx else 0
        sobel_dy = 0 if dx else 1
        self.apply_filter(
            Gradien_Sobel().apply_sobel,
            ddepth=cv2.CV_8U,
            dx=sobel_dx, dy=sobel_dy,
            ksize=ksize, scale=scale,
            original=self.original_image,
            blend=blend, blend_scale=blend_scale
        )
    def on_filter_result(self, result, method_name):
        """Xử lý kết quả filter, update display và plots"""
        if result is not None:
            self.transformed_image = result
            self.display_image(self.transformed_image, self.transformed_label)
            self.executor.submit(self.update_transformed_histogram, self.transformed_image)
            self.executor.submit(self.update_transformed_display, self.transformed_image)
            # Log nếu có steps (cho frequency filters)
            if hasattr(result, 'steps'):  # Frequency filters return tuple (img, steps)
                _, steps = result  # Giả sử func return tuple nếu verbose
                self.log_steps(steps)
        else:
            print(f"{method_name} failed.")

    def log_steps(self, steps):
        """Display timing steps in log"""
        self.log_text.clear()
        for step, duration in steps.items():
            self.log_text.append(f"{step}: {duration:.4f}s")

    def display_image(self, img_array, label):
        """Display image (tối ưu hóa)"""
        if img_array is None:
            return

        h, w = img_array.shape[:2]

        # Vectorize: Clip và cast một lần
        if img_array.dtype != np.uint8:
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        # Grayscale to RGB
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            bytes_per_line = 3 * w
        else:
            bytes_per_line = 3 * w

        # Display
        qt_img = QImage(img_array.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)

    def update_original_histogram(self, img_array):
        """Update original histogram thread-safe"""
        try:
            _, hist_norm = self.histogram_processor.normalized_Histogram(img_array)
            QMetaObject.invokeMethod(self.original_hist_canvas, "plot_histogram", 
                                     Qt.QueuedConnection, Q_ARG(np.ndarray, hist_norm), 
                                     Q_ARG(str, "Histogram - Original"))
        except Exception as e:
            print(f"Original histogram update error: {e}")

    def update_transformed_histogram(self, img_array):
        """Update transformed histogram thread-safe"""
        try:
            _, hist_norm = self.histogram_processor.normalized_Histogram(img_array)
            QMetaObject.invokeMethod(self.transformed_hist_canvas, "plot_histogram", 
                                     Qt.QueuedConnection, Q_ARG(np.ndarray, hist_norm), 
                                     Q_ARG(str, "Histogram - Transformed"))
        except Exception as e:
            print(f"Transformed histogram update error: {e}")

    def update_original_display(self, img_array):
        """Update original frequency plot thread-safe"""
        if img_array is None:
            return
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        try:
            freq = self.original_freq_canvas.compute_magnitude_spectrum(gray)
            QMetaObject.invokeMethod(self.original_freq_canvas, "plot_frequency", 
                                     Qt.QueuedConnection, Q_ARG(np.ndarray, freq), 
                                     Q_ARG(str, "DFT - Original"))
        except Exception as e:
            print(f"Original frequency update error: {e}")

    def update_transformed_display(self, img_array):
        """Update transformed frequency plot thread-safe"""
        if img_array is None:
            return
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        try:
            freq = self.transformed_freq_canvas.compute_magnitude_spectrum(gray)
            QMetaObject.invokeMethod(self.transformed_freq_canvas, "plot_frequency", 
                                     Qt.QueuedConnection, Q_ARG(np.ndarray, freq), 
                                     Q_ARG(str, "DFT - Transformed"))
        except Exception as e:
            print(f"Transformed frequency update error: {e}")

    def closeEvent(self, event):
        # Dừng executor
        self.executor.shutdown(wait=True)
        
        # Đợi tất cả QThread kết thúc
        for thread in self.active_threads[:]:
            if thread.isRunning():
                thread.quit()
                thread.wait(3000)  # Đợi tối đa 3s
        
        event.accept()