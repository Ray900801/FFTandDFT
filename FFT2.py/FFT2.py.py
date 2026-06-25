import numpy as np
import cv2
import matplotlib.pyplot as plt

def compress_color_fft_with_axes(image_path, keep_fraction=0.01):
    # 1. Load and prepare image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("Error: Could not read image.")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rows, cols, _ = img_rgb.shape
    
    # 2. Process Channels
    channels = cv2.split(img_rgb)
    compressed_channels = []
    
    # We'll create a master mask based on the Grayscale version for consistency
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    f_gray = np.fft.fftshift(np.fft.fft2(img_gray))
    magnitude_gray = np.abs(f_gray)
    
    # Create the threshold mask
    thresh = np.sort(magnitude_gray.ravel())[int(np.floor((1 - keep_fraction) * magnitude_gray.size))]
    mask = magnitude_gray > thresh
    
    # Process each color channel using the master mask
    for ch in channels:
        ch_fft = np.fft.fftshift(np.fft.fft2(ch))
        ch_compressed = ch_fft * mask # Apply compression
        
        # Inverse FFT
        ch_back = np.fft.ifft2(np.fft.ifftshift(ch_compressed))
        compressed_channels.append(np.clip(np.abs(ch_back), 0, 255).astype(np.uint8))
    
    img_reconstructed = cv2.merge(compressed_channels)
    
    # --- Visualization ---
    plt.figure(figsize=(20, 7))
    
    # 1. Original Plot (Spatial Domain)
    plt.subplot(131)
    plt.imshow(img_rgb)
    plt.title('Original Image', fontsize=12)
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.grid(False)

    # 2. Processed Frequency Space (Frequency Domain)
    plt.subplot(132)
    # 'extent' centers the axes at (0,0)
    extent = [-cols//2, cols//2, -rows//2, rows//2]
    # Use Log scale to make the frequencies visible
    spec_plot = plt.imshow(np.log(1 + magnitude_gray * mask), cmap='magma', extent=extent)
    plt.title(f'Processed Frequency Map\n(Kept top {keep_fraction*100}%)', fontsize=12)
    plt.xlabel('u (Horizontal Frequency)')
    plt.ylabel('v (Vertical Frequency)')
    plt.colorbar(spec_plot, fraction=0.046, pad=0.04, label='Log Magnitude')

    # 3. Reconstructed Plot (Spatial Domain)
    plt.subplot(133)
    plt.imshow(img_reconstructed)
    plt.title(f'Reconstructed Result\n({keep_fraction*100}% data)', fontsize=12)
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.grid(False)
    
    plt.tight_layout()
    plt.show()

# Run the updated program
# Using 0.01 (1%) will show a very clear "Star" pattern on the axes
compress_color_fft_with_axes('Yamada_Anna.jpg', keep_fraction=0.01)
