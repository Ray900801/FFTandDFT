import numpy as np
import cv2
import matplotlib.pyplot as plt

def compress_image_fft(image_path, keep_fraction=0.1):
    # 1. 以灰度模式读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Could not read image.")
        return

    # 2. 执行二维 FFT
    # 将图像转换到频域
    img_fft = np.fft.fft2(img)
    # 将零频分量移到频谱中心（方便处理）
    img_fft_shift = np.fft.fftshift(img_fft)
    
    # 3. 计算振幅谱（用于确定阈值）
    magnitude_spectrum = np.abs(img_fft_shift)
    
    # 4. 压缩逻辑：保留占比为 keep_fraction 的最强频率
    # 将所有振幅排序
    sorted_abs = np.sort(np.abs(img_fft_shift.reshape(-1)))
    # 找到对应比例的阈值
    thresh = sorted_abs[int(np.floor((1 - keep_fraction) * len(sorted_abs)))]
    
    # 创建掩码：振幅大于阈值的保留，其余置零
    mask = np.abs(img_fft_shift) > thresh
    img_fft_shift_compressed = img_fft_shift * mask
    
    # 5. 执行逆 FFT 恢复图像
    # 先移回原来的中心
    img_fft_ifftshift = np.fft.ifftshift(img_fft_shift_compressed)
    # 逆变换
    img_back = np.fft.ifft2(img_fft_ifftshift)
    # 取实部并限制在 0-255
    img_back = np.abs(img_back)

    # 6. 可视化结果
    plt.figure(figsize=(16, 8))
    
    plt.subplot(131), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.axis('off')
    
    plt.subplot(132), plt.imshow(np.log(1 + np.abs(img_fft_shift_compressed)), cmap='gray')
    plt.title(f'FFT (Keep {keep_fraction*100}%)'), plt.axis('off')
    
    plt.subplot(133), plt.imshow(img_back, cmap='gray')
    plt.title(f'Compressed Reconstruction'), plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 使用示例
# 替换为你的图片路径，keep_fraction 表示保留前 5% 的频率信息
compress_image_fft('maxresdefault.jpg', keep_fraction=0.05)
