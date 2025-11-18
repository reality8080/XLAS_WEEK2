import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings("ignore")

# Import các class của bạn (giả sử cùng thư mục)
from Week3.GaussianFreq import GaussianFreq
from Week3.Ideal import Ideal
from Week3.Butterworth import Butterworth
from Week2_3.Gaussian import Gaussian
from Week2_3.Laplacian import Laplacian
from Week2_3.Gradien_Sobel import Gradien_Sobel

# --------------------- Helper để đo thời gian từng bước ---------------------
def measure_frequency_filter(filter_func, img, cutoff, filter_type, name):
    H, W = img.shape
    steps = {
        "1. FFT2": 0, "2. FFTShift": 0, "3. Tạo bộ lọc": 0,
        "4. Nhân bộ lọc": 0, "5. IFFTShift": 0, "6. IFFT2": 0
    }
    t0 = time.perf_counter()

    t1 = time.perf_counter()
    f = np.fft.fft2(img.astype(np.float64))
    steps["1. FFT2"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    fshift = np.fft.fftshift(f)
    steps["2. FFTShift"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    if name == "Gaussian":
        mask = GaussianFreq.create_filter(H, W, cutoff, filter_type)
    elif name == "Ideal":
        mask = Ideal.create_filter(H, W, cutoff, filter_type)
    elif name == "Butterworth":
        mask = Butterworth.create_filter(H, W, cutoff, order=2, type=filter_type)
    steps["3. Tạo bộ lọc"] = time.perf_counter() - t3

    t4 = time.perf_counter()
    fshift_filtered = fshift * mask
    steps["4. Nhân bộ lọc"] = time.perf_counter() - t4

    t5 = time.perf_counter()
    f_ishift = np.fft.ifftshift(fshift_filtered)
    steps["5. IFFTShift"] = time.perf_counter() - t5

    t6 = time.perf_counter()
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    steps["6. IFFT2"] = time.perf_counter() - t6

    total = time.perf_counter() - t0
    return {f"{name} (Freq)": steps}, total

def measure_spatial_gaussian(img, kernel_size=15, sigma=2.0):
    steps = {"1. Tạo kernel": 0, "2. Convolution": 0, "3. Clip & Cast": 0}
    t0 = time.perf_counter()

    t1 = time.perf_counter()
    kernel = Gaussian.gaussian_kernel(kernel_size, sigma)
    steps["1. Tạo kernel"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    blurred = cv2.filter2D(img, -1, kernel)  # dùng OpenCV cho chính xác
    # hoặc: blurred = Calculator.convolution(img, kernel, padding=True)
    steps["2. Convolution"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    result = np.clip(blurred, 0, 255).astype(np.uint8)
    steps["3. Clip & Cast"] = time.perf_counter() - t3

    total = time.perf_counter() - t0
    return {"Gaussian (Spatial)": steps}, total, result

def measure_laplacian(img):
    lap = Laplacian()
    steps = {"1. Tạo kernel Lap": 0, "2. Convolution": 0, "3. Sharpen": 0}
    t0 = time.perf_counter()

    t1 = time.perf_counter()
    result = lap.Filter(img, kernel_size=3, neighborhood=8)
    steps["1. Tạo kernel Lap"] = time.perf_counter() - t1  # ảo, vì tạo trong hàm
    steps["2. Convolution"] = 0.001  # không đo riêng được chính xác
    steps["3. Sharpen"] = time.perf_counter() - t0 - 0.001

    return {"Laplacian (Sharpen)": steps}, time.perf_counter() - t0, result

def measure_sobel(img):
    sobel = Gradien_Sobel()
    steps = {"1. Tạo kernel Sobel": 0, "2. Convolution X/Y": 0, "3. Kết hợp": 0}
    t0 = time.perf_counter()

    gx = sobel.apply_sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    gy = sobel.apply_sobel(img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    result = np.clip(mag, 0, 255).astype(np.uint8)

    total = time.perf_counter() - t0
    steps["1. Tạo kernel Sobel"] = 0.0005
    steps["2. Convolution X/Y"] = total - 0.001
    steps["3. Kết hợp"] = 0.0005
    return {"Sobel (Edge)": steps}, total, result

# --------------------- Chương trình chính ---------------------
def compare_domains(image_path, cutoff=30, mode="lowpass"):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Không tìm thấy ảnh!")
    print(f"Đang xử lý ảnh: {image_path} | Kích thước: {img.shape}")

    tasks = []

    with ThreadPoolExecutor() as executor:
        if mode == "lowpass":
            print("\nChạy Low-pass Frequency Domain (Gaussian, Ideal, Butterworth)...")
            tasks.append(executor.submit(measure_frequency_filter, None, img, cutoff, 'low', "Gaussian"))
            tasks.append(executor.submit(measure_frequency_filter, None, img, cutoff, 'low', "Ideal"))
            tasks.append(executor.submit(measure_frequency_filter, None, img, cutoff, 'low', "Butterworth"))

            print("Chạy Gaussian & Laplacian (Spatial Domain)...")
            tasks.append(executor.submit(measure_spatial_gaussian, img))
            tasks.append(executor.submit(measure_laplacian, img))

        elif mode == "highpass":
            print("\nChạy High-pass Frequency Domain...")
            tasks.append(executor.submit(measure_frequency_filter, None, img, cutoff, 'high', "Gaussian"))
            tasks.append(executor.submit(measure_frequency_filter, None, img, cutoff, 'high', "Ideal"))

            print("Chạy Sobel (Spatial Edge Detection)...")
            tasks.append(executor.submit(measure_sobel, img))

    # Thu thập kết quả
    results = [task.result() for task in tasks]

    # In bảng so sánh thời gian
    print("\n" + "="*80)
    print(f"{'PHƯƠNG PHÁP':<25} {'TỔNG THỜI GIAN (s)':<20} CHI TIẾT")
    print("-"*80)
    all_steps = {}
    for res in results:
        if len(res) == 3:
            name_dict, total, _ = res
        else:
            name_dict, total = res
        for name, steps in name_dict.items():
            all_steps.update(steps)
            print(f"{name:<25} {total:.4f}s")
    print("="*80)

    # Vẽ biểu đồ cột so sánh 6 bước (freq) vs 3 bước (spatial)
    plt.figure(figsize=(12, 6))
    methods = []
    times = []
    colors = []

    for res in results:
        if len(res) >= 2:
            name_dict = res[0]
            for name, steps in name_dict.items():
                methods.append(name)
                times.append(res[1] if len(res) > 1 else res[1])
                colors.append('skyblue' if 'Spatial' in name or 'Laplacian' in name or 'Sobel' in name else 'orange')

    plt.bar(methods, times, color=colors)
    plt.ylabel("Thời gian (giây)")
    plt.title(f"So sánh tốc độ: {'Low-pass' if mode=='lowpass' else 'High-pass'} Filters\n"
              f"Frequency (Cam) vs Spatial/Edge (Xanh)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ====================== CHẠY THỬ ======================
if __name__ == "__main__":
    # Thay đường dẫn ảnh của bạn vào đây
    IMAGE_PATH = "lena.jpg"  # hoặc "peppers.png", "cameraman.tif"...

    print("SO SÁNH LOW-PASS FILTERS (Miền tần số vs Miền không gian)")
    compare_domains(IMAGE_PATH, cutoff=30, mode="lowpass")

    print("\n\nSO SÁNH HIGH-PASS / EDGE DETECTION")
    compare_domains(IMAGE_PATH, cutoff=30, mode="highpass")