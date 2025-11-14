# gui/image_app_gui.py

from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog,
    QHBoxLayout, QSlider, QGroupBox, QFormLayout
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import numpy as np

# Giả định HistogramPlot là một class riêng, cần được import
from .histogram_plot import HistogramPlot 

# Giả định các class xử lý ảnh đã được định nghĩa trong thư mục Class/
from Class.Negative import Negative
from Class.Histogram import Histogram
from Class.Log_Transform import Log_Transform
from Class.Gamma import Gamma
from Class.Piecewise_Linear_Transform import Piecewise_Linear_Transform
from Class.Box_Filter import Box_Filter


class ImageApp(QWidget):
    def __init__(self, 
                 negative_processor: Negative, 
                 log_processor: Log_Transform,
                 gamma_processor: Gamma, 
                 piecewise_processor:Piecewise_Linear_Transform, 
                 histogram_processor: Histogram,
                 box_filter_processor:Box_Filter
                 ):
        super().__init__()
        # === 1. Khởi tạo các Bộ xử lý ===
        self.negative_processor = negative_processor
        self.log_processor = log_processor
        self.gamma_processor = gamma_processor
        self.piecewise_processor = piecewise_processor
        self.histogram_processor = histogram_processor
        self.box_filter_processor = box_filter_processor

        # === 2. Trạng thái ảnh và chế độ tự động ===
        self.current_image_rgb = None
        self.original_gray = None
        self.use_auto_piecewise = True
        self.use_auto_log = True
        self.use_auto_gamma = True

        # === 3. Cài đặt Giao diện ===
        self.setWindowTitle("Bộ hiển thị ảnh số")
        self.setGeometry(100, 100, 1200, 700)
        
        self.setup_ui()
        self.connect_signals()
        
    def setup_ui(self):
        # === Khởi tạo các thành phần giao diện ===
        self.original_label = QLabel("Ảnh gốc")
        self.transformed_label = QLabel("Ảnh biến đổi")
        for label in [self.original_label, self.transformed_label]:
            label.setAlignment(Qt.AlignCenter)
            label.setFixedSize(400, 400)
            label.setStyleSheet("border: 1px solid gray; background: #f0f0f0;")
            
        self.original_hist_canvas = HistogramPlot(self)
        self.transformed_hist_canvas = HistogramPlot(self)
        self.original_hist_canvas.setFixedHeight(200)
        self.transformed_hist_canvas.setFixedHeight(200)

        # === 1. Layout Hiển thị Ảnh và Histogram ===
        original_group = QVBoxLayout()
        original_group.addWidget(self.original_label)
        original_group.addWidget(self.original_hist_canvas)

        transformed_group = QVBoxLayout()
        transformed_group.addWidget(self.transformed_label)
        transformed_group.addWidget(self.transformed_hist_canvas)

        images_layout = QHBoxLayout()
        images_layout.addLayout(original_group)
        images_layout.addLayout(transformed_group)

        # === 2. Khởi tạo các Widget Điều khiển ===
        self.load_button = QPushButton("Chọn ảnh từ máy")
        self.negative_button = QPushButton("Ảnh âm bản")
        self.log_button = QPushButton("Biến đổi Log")
        self.gamma_button = QPushButton("Biến đổi Gamma")
        self.piecewise_button = QPushButton("Biến đổi đoạn tuyến tính")
        self.reset_log_button = QPushButton("Reset Log")
        self.reset_gamma_button = QPushButton("Reset Gamma")
        self.reset_piecewise_button = QPushButton("Reset đoạn tuyến tính")
        self.hist_equal_button = QPushButton("Cân bằng Histogram (HE)")
        self.box_filter_button = QPushButton("Áp dụng Box Filter")

        # Khởi tạo Sliders (dùng hàm helper để giảm lặp)
        self.slider = self._create_slider(1, 255, 50, 5) # Log C
        self.gamma_slider = self._create_slider(10, 250, 100, 10) # Gamma (0.1 đến 2.5)
        self.kernel_slider = self._create_slider(3, 15, 3, 2, True) # Box Filter Kernel (chỉ số lẻ)
        
        self.r1_slider = self._create_slider(0, 255, 70, 1) # Piecewise
        self.s1_slider = self._create_slider(0, 255, 0, 1)
        self.r2_slider = self._create_slider(0, 255, 140, 1)
        self.s2_slider = self._create_slider(0, 255, 255, 1)
        
        # === 3. Layout Điều khiển ===
        control_layout = QFormLayout()
        
        control_layout.addRow(self.load_button)
        control_layout.addRow(self.negative_button)
        control_layout.addRow(self.log_button)
        control_layout.addRow(self.hist_equal_button)
        
        control_layout.addRow("Giá trị c (Log):", self.slider)
        control_layout.addRow(self.reset_log_button)

        control_layout.addRow("Giá trị gamma:", self.gamma_slider)
        control_layout.addRow(self.gamma_button)
        control_layout.addRow(self.reset_gamma_button)
        
        control_layout.addRow("r1:", self.r1_slider)
        control_layout.addRow("s1:", self.s1_slider)
        control_layout.addRow("r2:", self.r2_slider)
        control_layout.addRow("s2:", self.s2_slider)
        control_layout.addRow(self.piecewise_button)
        control_layout.addRow(self.reset_piecewise_button)

        control_layout.addRow("Kích thước Kernel:", self.kernel_slider)
        control_layout.addRow(self.box_filter_button)

        control_box = QGroupBox("Tùy chọn xử lý ảnh")
        control_box.setLayout(control_layout)
        control_box.setFixedWidth(350)
        
        # === 4. Layout Chính ===
        main_layout = QHBoxLayout()
        main_layout.addLayout(images_layout, stretch=1)
        main_layout.addWidget(control_box)
        self.setLayout(main_layout)

    def _create_slider(self, min_val, max_val, default_val, interval, is_odd_step=False):
        """Hàm tiện ích tạo QSlider."""
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default_val)
        slider.setTickInterval(interval)
        slider.setTickPosition(QSlider.TicksBelow)
        if is_odd_step:
            slider.setSingleStep(2)
        return slider

    def connect_signals(self):
        """Kết nối tất cả các sự kiện."""
        self.load_button.clicked.connect(self.load_image)
        self.negative_button.clicked.connect(self.apply_negative)
        self.log_button.clicked.connect(self.apply_log_transform)
        self.gamma_button.clicked.connect(self.apply_gamma_transform)
        self.piecewise_button.clicked.connect(self.apply_piecewise_transform)
        self.box_filter_button.clicked.connect(self.apply_box_filter)
        self.hist_equal_button.clicked.connect(self.apply_histogram_equalization)

        self.reset_log_button.clicked.connect(self.reset_log_slider)
        self.reset_gamma_button.clicked.connect(self.reset_gamma_slider)
        self.reset_piecewise_button.clicked.connect(self.reset_piecewise_slider)

        # Tắt chế độ auto khi người dùng kéo slider thủ công
        self.slider.valueChanged.connect(lambda: setattr(self, 'use_auto_log', False))
        self.gamma_slider.valueChanged.connect(lambda: setattr(self, 'use_auto_gamma', False))
        for slider in [self.r1_slider, self.s1_slider, self.r2_slider, self.s2_slider]:
            slider.valueChanged.connect(lambda: setattr(self, 'use_auto_piecewise', False))

    # --- Các phương thức Xử lý Ảnh ---

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Chọn ảnh", "", "Image Files (*.png *.jpg *.bmp *.jpeg)"
        )
        if not file_name:
            return

        img_bgr = cv2.imread(file_name)
        if img_bgr is None:
            return

        self.current_image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Tạo ảnh xám gốc để dùng cho các bộ lọc 2D
        self.original_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) 

        pixmap = QPixmap(file_name).scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.original_label.setPixmap(pixmap)
        self.transformed_label.setPixmap(pixmap)

        # CẬP NHẬT HISTOGRAM GỐC
        _, self.original_hist = self.histogram_processor.normalized_Histogram(
            self.current_image_rgb
        )
        self.original_hist_canvas.plot_histogram(self.original_hist, "Histogram - Ảnh gốc")

        # CẬP NHẬT HISTOGRAM BIẾN ĐỔI (ban đầu giống gốc)
        self.update_transformed_histogram(self.current_image_rgb)

    def apply_negative(self):
        if hasattr(self, 'current_image_rgb') and self.current_image_rgb is not None:
            self.negative_processor.image = self.current_image_rgb
            neg_img = self.negative_processor.negative()
            self.display_image(neg_img)

    def apply_log_transform(self):
        if hasattr(self, 'current_image_rgb') and self.current_image_rgb is not None:
            if self.use_auto_log:
                # Giả định log_transform() có thể nhận c=0 để tính c tự động
                log_img = self.log_processor.log_transform(self.current_image_rgb, 0, True)
            else:
                c_value = self.slider.value()
                log_img = self.log_processor.log_transform(self.current_image_rgb, c_value, False)
            self.display_image(log_img)

    def apply_gamma_transform(self):
        if hasattr(self, 'current_image_rgb') and self.current_image_rgb is not None:
            if self.use_auto_gamma:
                gamma_img, _ = self.gamma_processor.auto_gamma(self.current_image_rgb)
            else:
                gamma_value = self.gamma_slider.value() / 100.0
                gamma_img = self.gamma_processor.gamma_transform(self.current_image_rgb, gamma_value)
            self.display_image(gamma_img)
            
    def apply_piecewise_transform(self):
        if hasattr(self, 'current_image_rgb') and self.current_image_rgb is not None:
            if self.use_auto_piecewise:
                r1, r2 = self.piecewise_processor.auto_slicing_threshold(self.current_image_rgb)
                s1, s2 = 0, 255
            else:
                r1 = self.r1_slider.value()
                s1 = self.s1_slider.value()
                r2 = self.r2_slider.value()
                s2 = self.s2_slider.value()

            transformed = self.piecewise_processor.piecewise_linear_transform(self.current_image_rgb, r1, s1, r2, s2)
            self.display_image(transformed)

    def apply_histogram_equalization(self):
        if self.original_gray is not None:
            # 1. Tính toán Histogram đã chuẩn hóa của ảnh xám gốc
            img_gray, hist_norm = self.histogram_processor.normalized_Histogram(
                self.original_gray
            )
            
            # 2. Áp dụng Cân bằng Histogram
            img_eq_gray = self.histogram_processor.histogram_Equalization(
                img_gray, hist_norm
            )

            # 3. Chuyển ảnh xám đã cân bằng về định dạng RGB để hiển thị
            img_eq_rgb = cv2.cvtColor(img_eq_gray, cv2.COLOR_GRAY2RGB)

            # 4. Hiển thị ảnh
            self.display_image(img_eq_rgb)
        else:
            print("Vui lòng tải ảnh trước.")

    def apply_box_filter(self):
        if self.original_gray is not None:
            kernel_size = self.kernel_slider.value()
            
            # Box Filter chỉ áp dụng cho ảnh xám (self.original_gray)
            filtered_gray = self.box_filter_processor.filter(
                img=self.original_gray, 
                kernel_size=kernel_size, 
                padding=True
            )

            # Chuyển ảnh xám đã lọc về định dạng RGB để hiển thị
            filtered_img_rgb = cv2.cvtColor(filtered_gray, cv2.COLOR_GRAY2RGB)
            
            self.display_image(filtered_img_rgb)
        
    # --- Các phương thức Utility ---

    def reset_log_slider(self):
        self.use_auto_log = True
    def reset_gamma_slider(self):
        self.use_auto_gamma = True
    def reset_piecewise_slider(self):
        self.use_auto_piecewise = True

    def display_image(self, img_array):
        """Hiển thị ảnh và cập nhật histogram."""
        if img_array is None:
            return

        h, w = img_array.shape[:2]
        bytes_per_line = 3 * w
        
        # Đảm bảo dữ liệu là 8-bit unsigned integer (uchar)
        if img_array.dtype != np.uint8:
            img_array = img_array.astype(np.uint8)
            
        qt_img = QImage(img_array.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.transformed_label.setPixmap(pixmap)

        # CẬP NHẬT HISTOGRAM SAU BIẾN ĐỔI
        self.update_transformed_histogram(img_array)

    def update_transformed_histogram(self, img_array):
        _, hist_norm = self.histogram_processor.normalized_Histogram(img_array)
        self.transformed_hist_canvas.plot_histogram(hist_norm, "Histogram - Ảnh biến đổi")