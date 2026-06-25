'''
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from PIL import Image

# -----------------------------------------------------------------------
# 1. Brutal Matrix Fourier Transform Functions
# -----------------------------------------------------------------------
def get_dft_matrix(N):
    """Generates the classic N x N DFT Matrix brutally."""
    n = np.arange(N)
    k = n.reshape((N, 1))
    W = np.exp(-2j * np.pi * k * n / N)
    return W

def forward_dft_2d(channel, W_mat):
    """2D DFT using standard matrix multiplication: W @ channel @ W"""
    return W_mat @ channel @ W_mat

def inverse_dft_2d(X, W_mat):
    """2D IDFT using the conjugate transpose of the DFT matrix."""
    N = X.shape[0]
    W_inv = W_mat.conj().T / N
    return W_inv @ X @ W_inv

# -----------------------------------------------------------------------
# 2. File Loading via Input
# -----------------------------------------------------------------------
img_path = input("Enter the path to your image file (e.g., photo.jpg):").strip()

if not os.path.exists(img_path):
    print(f"File not found: '{img_path}'. Generating a colorful fallback pattern.")
    # Create synthetic canvas if file path is wrong
    raw_source = np.zeros((256, 256, 3))
    y, x = np.mgrid[0:256, 0:256]
    raw_source[:, :, 0] = (np.sin(x/10) * np.cos(y/10) + 1) / 2
    raw_source[:, :, 1] = (np.sin(x/20) + 1) / 2
    raw_source[:, :, 2] = (np.cos(x/15 + y/15) + 1) / 2
else:
    # Open image, enforce RGB mode, and normalize pixels to float [0, 1]
    pil_img = Image.open(img_path).convert('RGB')
    raw_source = np.array(pil_img) / 255.0

# -----------------------------------------------------------------------
# 3. Interactive Plot and System State Setup
# -----------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(14, 5))
plt.subplots_adjust(bottom=0.3)  # Room for two sliders

ax_raw, ax_freq, ax_proc = axs[0], axs[1], axs[2]

# Initial State Variables
init_N = 100 # 2^6 = 64x64
init_keep = 1.0

# Placeholders for our dynamic variables global to the update routine
current_dim = 2 ** init_N
img_resized = None
img_fft = None
W_mat = None

# Create visual display plots
raw_display = ax_raw.imshow(np.zeros((1, 1, 3)))
freq_display = ax_freq.imshow(np.zeros((1, 1)), cmap="twilight_shifted")
proc_display = ax_proc.imshow(np.zeros((1, 1, 3)))

for ax in axs:
    ax.axis("off")

# Add the interactive sliders
ax_slider_N = plt.axes([0.25, 0.12, 0.5, 0.03])
ax_slider_keep = plt.axes([0.25, 0.05, 0.5, 0.03])

# Slider N limits set safely up to 7 (128x128). Matrix Fourier over 128 is highly sluggish.
slider_N = Slider(ax=ax_slider_N, label='N (Dim = $2^N$)', valmin=4, valmax=100, valinit=init_N, valfmt='%d')
slider_keep = Slider(ax=ax_slider_keep, label='Keep Fraction', valmin=0.0, valmax=1.0, valinit=init_keep, valfmt='%1.2f')

# -----------------------------------------------------------------------
# 4. Processing Pipelines
# -----------------------------------------------------------------------
def compute_base_fft():
    """Resizes raw image to 2^N and computes explicit matrix DFT transforms."""
    global img_resized, img_fft, W_mat, current_dim
    
    N_val = int(slider_N.val)
    current_dim = 2 ** N_val
    
    # Process image scaling cleanly using PIL to prevent numpy aliasing artifacts
    pil_src = Image.fromarray((raw_source * 255).astype(np.uint8))
    pil_resized = pil_src.resize((current_dim, current_dim), Image.Resampling.LANCZOS)
    img_resized = np.array(pil_resized) / 255.0
    
    # Generate the brutal NxN Fourier transform matrix
    W_mat = get_dft_matrix(current_dim)
    
    # Run the matrix transformation over R, G, B channels
    img_fft = np.zeros((current_dim, current_dim, 3), dtype=complex)
    for c in range(3):
        img_fft[:, :, c] = forward_dft_2d(img_resized[:, :, c], W_mat)
        
    # Update raw view layout boundaries
    ax_raw.set_title(f"Raw Image ({current_dim}x{current_dim})")
    raw_display.set_data(img_resized)
    raw_display.set_extent([0, current_dim, current_dim, 0])

def render_compression():
    """Filters frequencies based on keep fraction and executes IDFT."""
    keep_fraction = slider_keep.val
    
    compressed_fft = img_fft.copy()
    
    # Determine frequency cutoffs from corners
    bound = int(current_dim * keep_fraction / 2)
    
    # Construct lower boundary pass mask
    mask = np.zeros((current_dim, current_dim), dtype=bool)
    if bound > 0:
        mask[:bound, :bound] = True
        mask[-bound:, :bound] = True
        mask[:bound, -bound:] = True
        mask[-bound:, -bound:] = True
    elif keep_fraction > 0:
        mask[0, 0] = True

    # Drop frequencies falling outside the boundary box
    compressed_fft[~mask] = 0

    # Matrix IDFT reconstruct back to space
    reconstructed = np.zeros_like(img_resized)
    for c in range(3):
        channel_res = inverse_dft_2d(compressed_fft[:, :, c], W_mat)
        reconstructed[:, :, c] = np.clip(channel_res.real, 0, 1)

    # Re-render UI axes mapping variables safely
    ax_proc.set_title(f"Processed Image (Keep: {keep_fraction*100:.0f}%)")
    proc_display.set_data(reconstructed)
    proc_display.set_extent([0, current_dim, current_dim, 0])
    
    # Render log spectral footprint shifted cleanly to the display center
    updated_shifted = np.fft.fftshift(compressed_fft, axes=(0, 1))
    magnitude_spectrum = np.log(1 + np.abs(updated_shifted).mean(axis=-1))
    freq_display.set_data(magnitude_spectrum)
    freq_display.set_extent([0, current_dim, current_dim, 0])
    
    # Adjust dynamic plotting display bounds automatically
    for ax in axs:
        ax.relim()
        ax.autoscale_view()
        
    fig.canvas.draw_idle()

# -----------------------------------------------------------------------
# 5. UI Event Hooks
# -----------------------------------------------------------------------
def on_change_N(val):
    # Step integer restriction handling for the slider input
    discrete_N = int(slider_N.val)
    if slider_N.val != discrete_N:
        slider_N.set_val(discrete_N)
        return
    compute_base_fft()
    render_compression()

def on_change_keep(val):
    render_compression()

slider_N.on_changed(on_change_N)
slider_keep.on_changed(on_change_keep)

# Initialize pipeline values
compute_base_fft()
render_compression()

plt.show()

'''''

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from PIL import Image

# Automatically look for your uploaded forest image
image_name = "YamadaAnna.png"

# -----------------------------------------------------------------------------
# 1. High-Performance Iterative Cooley-Tukey Radix-2 FFT Implementation
# -----------------------------------------------------------------------------
def bit_reverse_copy(x):
    """Permutes the array indices by reversing their binary representations."""
    N = len(x)
    n = int(np.log2(N))
    # Generate bit-reversed indices
    idx = np.arange(N)
    rev_idx = np.zeros_like(idx)
    for i in range(n):
        rev_idx |= ((idx >> i) & 1) << (n - 1 - i)
    return x[rev_idx]

def fft_1d_iterative(x, inverse=False):
    """Iterative 1D Radix-2 FFT. Vastly faster than deep recursion in Python."""
    N = len(x)
    A = bit_reverse_copy(np.array(x, dtype=complex))
    
    n = int(np.log2(N))
    for s in range(1, n + 1):
        m = 1 << s
        # Calculate twiddle factors mathematically
        direction = 2j if inverse else -2j
        wm = np.exp(direction * np.pi / m)
        
        # Vectorized butterfly operations along blocks
        w = 1.0
        for j in range(m // 2):
            idx1 = np.arange(j, N, m)
            idx2 = idx1 + m // 2
            
            t = w * A[idx2]
            u = A[idx1]
            A[idx1] = u + t
            A[idx2] = u - t
            w *= wm
            
    if inverse:
        A /= N
    return A

def fft_2d(img_2d):
    """2D FFT using high-performance row-column 1D iterative transformations."""
    rows_fft = np.array([fft_1d_iterative(row) for row in img_2d])
    return np.array([fft_1d_iterative(col) for col in rows_fft.T]).T

def ifft_2d(freq_2d):
    """2D IFFT using high-performance row-column 1D iterative transformations."""
    rows_ifft = np.array([fft_1d_iterative(row, inverse=True) for row in freq_2d])
    return np.array([fft_1d_iterative(col, inverse=True) for col in rows_ifft.T]).T

# -----------------------------------------------------------------------------
# 2. Image Processing & Compression Pipeline
# -----------------------------------------------------------------------------
def process_channel(channel, N, keep_fraction):
    size = 2**N
    
    # Resize channel to match our 2^N grid
    img_pil = Image.fromarray(channel)
    img_resized = np.array(img_pil.resize((size, size), Image.Resampling.BILINEAR))
    
    # Forward 2D FFT
    freq = fft_2d(img_resized)
    freq_shifted = np.fft.fftshift(freq) 
    
    # Create Low-Pass Mask based on keep_fraction
    row, col = freq_shifted.shape
    crow, ccol = row // 2, col // 2
    mask = np.zeros((row, col))
    
    r = int((row // 2) * keep_fraction)
    if r == 0 and keep_fraction > 0:
        r = 1
        
    mask[crow-r:crow+r+1, ccol-r:ccol+r+1] = 1
    
    # Apply mask and unshift frequencies
    freq_shifted_compressed = freq_shifted * mask
    freq_compressed = np.fft.ifftshift(freq_shifted_compressed)
    
    # Inverse 2D FFT to reconstruct details
    reconstructed = np.real(ifft_2d(freq_compressed))
    reconstructed = np.clip(reconstructed, 0, 255) 
    
    mag_spectrum = np.log(1 + np.abs(freq_shifted))
    return reconstructed.astype(np.uint8), mag_spectrum

# -----------------------------------------------------------------------------
# 3. Image Loading Logic
# -----------------------------------------------------------------------------
if os.path.exists(image_name):
    try:
        img_raw = np.array(Image.open(image_name).convert('RGB'))
        print(f"Loaded '{image_name}' successfully.")
    except Exception as e:
        print(f"Error reading image file: {e}. Generating placeholder fallback.")
        image_name = None
else:
    print(f"Could not find '{image_name}'. Using a synthetic fallback pattern.")
    y, x = np.ogrid[-100:100, -100:100]
    img_raw = np.stack([np.uint8(np.clip(x**2 + y**2, 0, 255)), 
                        np.uint8(np.clip(100 + x*2, 0, 255)), 
                        np.uint8(np.clip(100 + y*2, 0, 255))], axis=-1)

# -----------------------------------------------------------------------------
# 4. Interactive UI Setup (Fixed N = 9 for 512x512 High Fidelity)
# -----------------------------------------------------------------------------
HIGH_RES_N = 9  # 2^9 = 512x512 matrix sizing

fig, (ax_raw, ax_freq, ax_proc) = plt.subplots(1, 3, figsize=(15, 6))
plt.subplots_adjust(bottom=0.2)

ax_keep = plt.axes([0.25, 0.05, 0.5, 0.03])
slider_keep = Slider(ax_keep, 'Keep Fraction', 0.0, 1.0, valinit=0.2, valstep=0.01)

def update(val):
    keep_frac = slider_keep.val
    size = 2**HIGH_RES_N
    
    recon_channels = []
    mag_channels = []
    
    # Process RGB channels independently
    for i in range(3):
        recon_ch, mag_ch = process_channel(img_raw[:,:,i], HIGH_RES_N, keep_frac)
        recon_channels.append(recon_ch)
        mag_channels.append(mag_ch)
        
    img_processed = np.stack(recon_channels, axis=-1)
    img_freq = np.mean(mag_channels, axis=0)
    
    img_raw_pil = Image.fromarray(img_raw).resize((size, size))
    
    # Render layout
    ax_raw.clear()
    ax_raw.imshow(img_raw_pil)
    ax_raw.set_title(f"Raw Image ({size}x{size})")
    ax_raw.axis('off')
    
    ax_freq.clear()
    ax_freq.imshow(img_freq, cmap='viridis')  # Viridis handles organic ranges beautifully
    ax_freq.set_title("Forest Frequency Space")
    ax_freq.axis('off')
    
    ax_proc.clear()
    ax_proc.imshow(img_processed)
    ax_proc.set_title(f"Compressed (Keep: {keep_frac:.2f})")
    ax_proc.axis('off')
    
    fig.canvas.draw_idle()

slider_keep.on_changed(update)
update(None)
plt.show()