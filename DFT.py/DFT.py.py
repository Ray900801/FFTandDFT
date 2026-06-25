import numpy as np
import cv2
import matplotlib.pyplot as plt

def compress_color_dft_demonstration(image_path, keep_fraction=0.01):
    # 1. Load image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("Error: Image not found.")
        return
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rows, cols, _ = img_rgb.shape
    
    # Split into R, G, B
    channels = cv2.split(img_rgb)
    reconstructed_channels = []
    
    # We will use the Luminance (Y) for the Frequency plot to represent "General" frequency
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # --- Process Frequency Space for Visualization ---
    f_transform = np.fft.fft2(img_gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    
    # Create the compression mask based on the keep_fraction
    thresh = np.sort(magnitude_spectrum.ravel())[int(np.floor((1 - keep_fraction) * magnitude_spectrum.size))]
    mask = magnitude_spectrum > thresh
    
    # Apply mask to the visualization spectrum
    processed_spectrum = np.log(1 + magnitude_spectrum * mask)

    # --- Process actual Color Channels ---
    for ch in channels:
        ch_fft = np.fft.fft2(ch)
        ch_shift = np.fft.fftshift(ch_fft)
        
        # Apply the SAME mask to all color channels to keep it consistent
        ch_compressed = ch_shift * mask
        
        # Inverse Shift and Inverse FFT
        ch_ishift = np.fft.ifftshift(ch_compressed)
        ch_back = np.fft.ifft2(ch_ishift)
        reconstructed_channels.append(np.clip(np.abs(ch_back), 0, 255).astype(np.uint8))

    img_output = cv2.merge(reconstructed_channels)

    # --- Visualization with Axes ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # Plot 1: Original
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Color Image", fontsize=14)
    axes[0].set_xlabel("Width (Pixels)")
    axes[0].set_ylabel("Height (Pixels)")

    # Plot 2: Processed Frequency Space
    # We use extent to show the frequency coordinates relative to the center (0,0)
    extent = [-cols//2, cols//2, -rows//2, rows//2]
    im2 = axes[1].imshow(processed_spectrum, cmap='viridis', extent=extent)
    axes[1].set_title(f"Processed Frequency Space\n(Top {keep_fraction*100}% Frequencies)", fontsize=14)
    axes[1].set_xlabel("u (Horizontal Frequency)")
    axes[1].set_ylabel("v (Vertical Frequency)")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Plot 3: Reconstructed
    axes[2].imshow(img_output)
    axes[2].set_title(f"Reconstructed\n(Data kept: {keep_fraction*100}%)", fontsize=14)
    axes[2].set_xlabel("Width (Pixels)")
    axes[2].set_ylabel("Height (Pixels)")

    plt.tight_layout()
    plt.show()

# Run the program
# Set keep_fraction to 0.01 to see the 1% 'Star' pattern with axes
compress_color_dft_demonstration('Yamada_Anna.jpg', keep_fraction=0.01)
