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
from .FrequencyPlot import FrequencyPlot

# Giả định các class xử lý ảnh đã được định nghĩa trong thư mục Week2_3/
from Week2_3.Negative import Negative
from Week2_3.Histogram import Histogram
from Week2_3.Log_Transform import Log_Transform
from Week2_3.Gamma import Gamma
from Week2_3.Piecewise_Linear_Transform import Piecewise_Linear_Transform
from Week2_3.Box_Filter import Box_Filter
from Week2_3.Meadian_Filter import Meadian_Filter # Lỗi chính tả trong tên file
from Week2_3.Max_Min_Filter import Max_Filter, Min_Filter, Mid_Filter
from Week2_3.Gaussian import Gaussian

from Week2_3.UnsharpMasking_HighBoost import UnsharpMasking_HighBoost
from Week2_3.Laplacian import Laplacian
from Week2_3.Gradien_Sobel import Gradien_Sobel

class ImageApp(QWidget):
    def __init__(self, 
                 negative_processor: Negative, 
                 log_processor: Log_Transform,
                 gamma_processor: Gamma, 
                 piecewise_processor:Piecewise_Linear_Transform, 
                 histogram_processor: Histogram,
                 box_filter_processor:Box_Filter,
                 median_filter_processor: Meadian_Filter,
                 max_filter_processor: Max_Filter,
                 min_filter_processor: Min_Filter,
                 mid_filter_processor: Mid_Filter,
                 gaussian_filter_processor: Gaussian,
                 unsharp_processor: UnsharpMasking_HighBoost,
                 laplacian_processor: Laplacian,
                 sobel_processor: Gradien_Sobel,
                 ):
        super().__init__()
        # === 1. Khởi tạo các Bộ xử lý ===
        self.negative_processor = negative_processor
        self.log_processor = log_processor
        self.gamma_processor = gamma_processor
        self.piecewise_processor = piecewise_processor
        self.histogram_processor = histogram_processor
        self.box_filter_processor = box_filter_processor
        self.median_filter_processor = median_filter_processor
        self.max_filter_processor = max_filter_processor
        self.min_filter_processor = min_filter_processor
        self.mid_filter_processor = mid_filter_processor
        self.gaussian_filter_processor = gaussian_filter_processor

        self.unsharp_processor = unsharp_processor
        self.laplacian_processor = laplacian_processor
        self.sobel_processor = sobel_processor

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

        self.original_freq_canvas = FrequencyPlot(self)
        self.transformed_freq_canvas = FrequencyPlot(self)
        self.original_freq_canvas.setFixedHeight(200)
        self.transformed_freq_canvas.setFixedHeight(200)

        # === 1. Layout Hiển thị Ảnh và Histogram ===
        original_group = QVBoxLayout()
        original_group.addWidget(self.original_label)
        original_group.addWidget(self.original_hist_canvas)
        original_group.addWidget(self.original_freq_canvas)

        transformed_group = QVBoxLayout()
        transformed_group.addWidget(self.transformed_label)
        transformed_group.addWidget(self.transformed_hist_canvas)
        transformed_group.addWidget(self.transformed_freq_canvas)

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
        self.median_filter_button = QPushButton("Áp dụng Median Filter")
        self.max_filter_button = QPushButton("Áp dụng Max Filter")
        self.min_filter_button = QPushButton("Áp dụng Min Filter")
        self.mid_filter_button = QPushButton("Áp dụng Mid Filter")
        self.gaussian_filter_button = QPushButton("Áp dụng Gaussian Filter")
        # Khởi tạo Sliders (dùng hàm helper để giảm lặp)
        self.slider = self._create_slider(1, 255, 50, 5) # Log C
        self.gamma_slider = self._create_slider(10, 250, 100, 10) # Gamma (0.1 đến 2.5)
        self.kernel_slider = self._create_slider(3, 15, 3, 2, True) # Box Filter Kernel (chỉ số lẻ)
        self.sigma_slider = self._create_slider(10, 50, 10, 1) # Sigma (0.1 đến 5.0)

        self.r1_slider = self._create_slider(0, 255, 70, 1) # Piecewise
        self.s1_slider = self._create_slider(0, 255, 0, 1)
        self.r2_slider = self._create_slider(0, 255, 140, 1)
        self.s2_slider = self._create_slider(0, 255, 255, 1)
        

        # --- Unsharp Masking ---
        self.unsharp_button = QPushButton("Unsharp Masking")
        self.highboost_button = QPushButton("High Boost")
        self.amount_slider = self._create_slider(0, 300, 100, 10)  # 0.0 -> 3.0
        self.threshold_slider = self._create_slider(0, 50, 0, 1)   # ngưỡng

        # --- Laplacian ---
        self.laplacian_button = QPushButton("Laplacian Filter")
        self.lap_neigh_slider = self._create_slider(0, 3, 1, 1)  # 0:4, 1:8, 2:16, 3:full

        # --- Sobel ---
        self.sobel_x_button = QPushButton("Sobel X")
        self.sobel_y_button = QPushButton("Sobel Y")
        self.sobel_mag_button = QPushButton("Sobel Magnitude")
        self.sobel_ksize_slider = self._create_slider(3, 15, 3, 2, True)  # chỉ số lẻ
        # === 3. Layout Điều khiển ===
        control_vbox = QVBoxLayout() # Vertical Box cho tất cả Group Boxes

# --- A. Tải Ảnh & Cơ bản ---
        load_group = QGroupBox("1. Tải Ảnh & Thao tác cơ bản")
        load_layout = QVBoxLayout()
        load_layout.addWidget(self.load_button)
        load_layout.addWidget(self.negative_button)
        load_layout.addWidget(self.hist_equal_button)
        load_group.setLayout(load_layout)
        control_vbox.addWidget(load_group)

        # --- B. Biến đổi Độ sáng (Log, Gamma, Piecewise) ---
        intensity_group = QGroupBox("2. Biến đổi Độ sáng (Intensity)")
        intensity_layout = QFormLayout()

        # --- Log Transform ---
        intensity_layout.addRow(self.log_button)
        intensity_layout.addRow("Giá trị c (Log):", self.slider)
        intensity_layout.addRow(self.reset_log_button)
        intensity_layout.addRow(QLabel("")) # Spacer

        # --- Gamma Transform ---
        intensity_layout.addRow(self.gamma_button)
        intensity_layout.addRow("Giá trị gamma:", self.gamma_slider)
        intensity_layout.addRow(self.reset_gamma_button)
        intensity_layout.addRow(QLabel("")) # Spacer

        # --- Piecewise Linear Transform ---
        intensity_layout.addRow(self.piecewise_button)
        intensity_layout.addRow("r1:", self.r1_slider)
        intensity_layout.addRow("s1:", self.s1_slider)
        intensity_layout.addRow("r2:", self.r2_slider)
        intensity_layout.addRow("s2:", self.s2_slider)
        intensity_layout.addRow(self.reset_piecewise_button)

        intensity_group.setLayout(intensity_layout)
        control_vbox.addWidget(intensity_group)

        # --- C. Lọc Không gian: Làm mờ & Giảm nhiễu (Smoothing) ---
        smoothing_group = QGroupBox("3. Lọc Giảm nhiễu (Smoothing)")
        smoothing_layout = QFormLayout()

        # Kernel Size Slider (dùng chung cho Box, Median, Max, Min, Mid, Gaussian)
        smoothing_layout.addRow("Kích thước Kernel lẻ:", self.kernel_slider)

        # Box Filter
        smoothing_layout.addRow(self.box_filter_button)

        # Gaussian Filter (Có thêm Sigma)
        smoothing_layout.addRow(self.gaussian_filter_button)
        smoothing_layout.addRow("Sigma (Gaussian):", self.sigma_slider)

        # Order-Statistic Filters (Dùng QHBoxLayout để gom 4 nút lại 1 hàng)
        order_layout = QHBoxLayout()
        order_layout.addWidget(self.median_filter_button)
        order_layout.addWidget(self.max_filter_button)
        order_layout.addWidget(self.min_filter_button)
        order_layout.addWidget(self.mid_filter_button)
        smoothing_layout.addRow("Lọc thống kê (Median/Max/Min/Mid):", order_layout)

        smoothing_group.setLayout(smoothing_layout)
        control_vbox.addWidget(smoothing_group)


        # --- D. Lọc Không gian: Làm sắc nét & Phát hiện cạnh (Sharpening/Edge) ---
        sharpen_group = QGroupBox("4. Lọc Sắc nét & Phát hiện cạnh")
        sharpen_layout = QFormLayout()

        # --- Unsharp Masking / High Boost ---
        sharpen_layout.addRow(self.unsharp_button)
        sharpen_layout.addRow(self.highboost_button)
        sharpen_layout.addRow("Amount:", self.amount_slider)
        sharpen_layout.addRow("Threshold:", self.threshold_slider)
        sharpen_layout.addRow(QLabel("")) # Spacer

        # --- Laplacian ---
        sharpen_layout.addRow(self.laplacian_button)
        sharpen_layout.addRow("Laplacian Neighbors:", self.lap_neigh_slider)
        sharpen_layout.addRow(QLabel("")) # Spacer

        # --- Sobel ---
        sharpen_layout.addRow("Sobel Kernel Size:", self.sobel_ksize_slider)
        sobel_h_layout = QHBoxLayout()
        sobel_h_layout.addWidget(self.sobel_x_button)
        sobel_h_layout.addWidget(self.sobel_y_button)
        sobel_h_layout.addWidget(self.sobel_mag_button)
        sharpen_layout.addRow("Sobel (X/Y/Mag):", sobel_h_layout)

        sharpen_group.setLayout(sharpen_layout)
        control_vbox.addWidget(sharpen_group)

        # --- Bọc tất cả Group Box vào QWidget có Scroll Bar (Đề xuất) ---
        # Do số lượng điều khiển nhiều, nên bọc lại để đảm bảo màn hình nhỏ vẫn cuộn được.
        scroll_widget = QWidget()
        scroll_widget.setLayout(control_vbox)

        from PyQt5.QtWidgets import QScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setFixedWidth(800) # Đặt chiều rộng cố định cho khu vực điều khiển
        
        # === 4. Layout Chính ===
        main_layout = QHBoxLayout()
        main_layout.addLayout(images_layout, stretch=1)
        main_layout.addWidget(scroll_area) # Thay thế control_box bằng scroll_area
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

        self.median_filter_button.clicked.connect(self.apply_median_filter)
        self.max_filter_button.clicked.connect(self.apply_max_filter)
        self.min_filter_button.clicked.connect(self.apply_min_filter)
        self.mid_filter_button.clicked.connect(self.apply_mid_filter)
        self.gaussian_filter_button.clicked.connect(self.apply_gaussian_filter)

        self.reset_log_button.clicked.connect(self.reset_log_slider)
        self.reset_gamma_button.clicked.connect(self.reset_gamma_slider)
        self.reset_piecewise_button.clicked.connect(self.reset_piecewise_slider)

        # Tắt chế độ auto khi người dùng kéo slider thủ công
        self.slider.valueChanged.connect(lambda: setattr(self, 'use_auto_log', False))
        self.gamma_slider.valueChanged.connect(lambda: setattr(self, 'use_auto_gamma', False))
        for slider in [self.r1_slider, self.s1_slider, self.r2_slider, self.s2_slider]:
            slider.valueChanged.connect(lambda: setattr(self, 'use_auto_piecewise', False))

        self.unsharp_button.clicked.connect(self.apply_unsharp)
        self.highboost_button.clicked.connect(self.apply_highboost)
        self.laplacian_button.clicked.connect(self.apply_laplacian)
        self.sobel_x_button.clicked.connect(self.apply_sobel_x)
        self.sobel_y_button.clicked.connect(self.apply_sobel_y)
        self.sobel_mag_button.clicked.connect(self.apply_sobel_magnitude)

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
        self.original_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        pixmap = QPixmap(file_name).scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.original_label.setPixmap(pixmap)
        self.transformed_label.setPixmap(pixmap)

        # Cập nhật histogram gốc
        _, self.original_hist = self.histogram_processor.normalized_Histogram(self.current_image_rgb)
        self.original_hist_canvas.plot_histogram(self.original_hist, "Histogram - Ảnh gốc")

        # Cập nhật frequency gốc
        freq_orig = self.original_freq_canvas.compute_magnitude_spectrum(self.original_gray)
        self.original_freq_canvas.plot_frequency(freq_orig, "DFT - Ảnh gốc")

        # Ban đầu: ảnh biến đổi = ảnh gốc
        self.display_image(self.current_image_rgb)

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
        
    def apply_median_filter(self):
        if self.original_gray is not None:
            kernel_size = self.kernel_slider.value()
            
            filtered_gray = self.median_filter_processor.Filter(
                image=self.original_gray, 
                kernel_size=kernel_size, 
                padding=True
            )
            filtered_img_rgb = cv2.cvtColor(filtered_gray, cv2.COLOR_GRAY2RGB)
            self.display_image(filtered_img_rgb)
        else:
            print("Vui lòng tải ảnh trước.")

    # THÊM PHƯƠNG THỨC CHO MAX FILTER
    def apply_max_filter(self):
        if self.original_gray is not None:
            kernel_size = self.kernel_slider.value()
            
            filtered_gray = self.max_filter_processor.Filter(
                image=self.original_gray.copy(), # Dùng .copy() để tránh thay đổi ảnh gốc
                kernel_size=kernel_size, 
                padding=True
            )
            filtered_img_rgb = cv2.cvtColor(filtered_gray, cv2.COLOR_GRAY2RGB)
            self.display_image(filtered_img_rgb)
        else:
            print("Vui lòng tải ảnh trước.")

    # THÊM PHƯƠNG THỨC CHO MIN FILTER
    def apply_min_filter(self):
        if self.original_gray is not None:
            kernel_size = self.kernel_slider.value()
            
            filtered_gray = self.min_filter_processor.Filter(
                image=self.original_gray.copy(), # Dùng .copy() để tránh thay đổi ảnh gốc
                kernel_size=kernel_size, 
                padding=True
            )
            filtered_img_rgb = cv2.cvtColor(filtered_gray, cv2.COLOR_GRAY2RGB)
            self.display_image(filtered_img_rgb)
        else:
            print("Vui lòng tải ảnh trước.")
            
    # THÊM PHƯƠNG THỨC CHO MIDPOINT FILTER
    def apply_mid_filter(self):
        if self.original_gray is not None:
            kernel_size = self.kernel_slider.value()
            
            filtered_gray = self.mid_filter_processor.Filter(
                image=self.original_gray.copy(), # Dùng .copy() để tránh thay đổi ảnh gốc
                kernel_size=kernel_size, 
                padding=True
            )
            filtered_img_rgb = cv2.cvtColor(filtered_gray, cv2.COLOR_GRAY2RGB)
            self.display_image(filtered_img_rgb)
        else:
            print("Vui lòng tải ảnh trước.")

    # THÊM PHƯƠNG THỨC CHO GAUSSIAN FILTER
    def apply_gaussian_filter(self):
        if self.original_gray is not None:
            kernel_size = self.kernel_slider.value()
            # Lấy giá trị sigma (0.1 đến 5.0)
            sigma_value = self.sigma_slider.value() / 10.0
            
            filtered_gray = self.gaussian_filter_processor.filter(
                img=self.original_gray, 
                kernel_size=kernel_size, 
                sigma=sigma_value,
                padding=True
            )

            filtered_img_rgb = cv2.cvtColor(filtered_gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            
            self.display_image(filtered_img_rgb)
        else:
            print("Vui lòng tải ảnh trước.")
    def apply_unsharp(self):
        if self.original_gray is not None:
            kernel_size = self.kernel_slider.value()
            sigma = self.sigma_slider.value() / 10.0
            amount = self.amount_slider.value() / 100.0
            threshold = self.threshold_slider.value()

            filtered = self.unsharp_processor.Filter(
                image=self.original_gray,
                kernel_size=kernel_size,
                sigma=sigma,
                amount=amount,
                threshold=threshold
            )
            filtered_rgb = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)
            self.display_image(filtered_rgb)

    def apply_highboost(self):
        if self.original_gray is not None:
            kernel_size = self.kernel_slider.value()
            sigma = self.sigma_slider.value() / 10.0
            amount = self.amount_slider.value() / 100.0 + 1.0  # High-boost: amount > 1
            threshold = 0

            filtered = self.unsharp_processor.Filter(
                image=self.original_gray,
                kernel_size=kernel_size,
                sigma=sigma,
                amount=amount,
                threshold=threshold
            )
            filtered_rgb = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)
            self.display_image(filtered_rgb)

    def apply_laplacian(self):
        if self.original_gray is not None:
            kernel_size = self.kernel_slider.value()
            neigh_idx = self.lap_neigh_slider.value()
            neigh_map = {0: 4, 1: 8, 2: 16, 3: kernel_size*kernel_size-1}
            neighborhood = neigh_map[neigh_idx]

            filtered = self.laplacian_processor.Filter(
                image=self.original_gray,
                kernel_size=kernel_size,
                neighborhood=neighborhood,
                padding=True
            )
            filtered_rgb = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)
            self.display_image(filtered_rgb)

    def apply_sobel_x(self):
        self._apply_sobel(dx=1, dy=0)

    def apply_sobel_y(self):
        self._apply_sobel(dx=0, dy=1)

    def apply_sobel_magnitude(self):
        self._apply_sobel(dx=1, dy=1)  # dùng cả 2 để tính magnitude

    def _apply_sobel(self, dx, dy):
        if self.original_gray is None:
            return

        ksize = self.sobel_ksize_slider.value()

        if dx == 1 and dy == 1:  # Magnitude
            gx = self.sobel_processor.apply_sobel(self.original_gray, cv2.CV_32F, 1, 0, ksize)
            gy = self.sobel_processor.apply_sobel(self.original_gray, cv2.CV_32F, 0, 1, ksize)
            mag = np.sqrt(gx**2 + gy**2)
            mag = np.clip(mag, 0, 255).astype(np.uint8)
            result = mag
        else:
            result = self.sobel_processor.apply_sobel(self.original_gray, cv2.CV_8U, dx, dy, ksize)

        result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        self.display_image(result_rgb)
    # --- Các phương thức Utility ---

    def reset_log_slider(self):
        self.use_auto_log = True
    def reset_gamma_slider(self):
        self.use_auto_gamma = True
    def reset_piecewise_slider(self):
        self.use_auto_piecewise = True

    def display_image(self, img_array):
        """Chỉ hiển thị ảnh và cập nhật histogram + frequency (qua hàm riêng)"""
        if img_array is None:
            return

        h, w = img_array.shape[:2]

        # Chuẩn hóa dữ liệu
        if img_array.dtype != np.uint8:
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        # Chuyển grayscale → RGB nếu cần
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            bytes_per_line = 3 * w
        else:
            bytes_per_line = 3 * w

        # Hiển thị ảnh
        qt_img = QImage(img_array.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.transformed_label.setPixmap(pixmap)

        # Cập nhật histogram
        self.update_transformed_histogram(img_array)

        # Cập nhật frequency (DFT) - KHÔNG gọi display_image
        self.update_transformed_display(img_array)  # ← an toàn, không lặp

    def update_transformed_histogram(self, img_array):
        _, hist_norm = self.histogram_processor.normalized_Histogram(img_array)
        self.transformed_hist_canvas.plot_histogram(hist_norm, "Histogram - Ảnh biến đổi")
    def update_transformed_display(self, img_array):
        """Chỉ cập nhật Frequency Plot (DFT), KHÔNG gọi display_image"""
        if img_array is None:
            return

        # Chuyển về ảnh xám
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Cập nhật phổ tần số
        freq_trans = self.transformed_freq_canvas.compute_magnitude_spectrum(gray)
        self.transformed_freq_canvas.plot_frequency(freq_trans, "DFT - Ảnh biến đổi")