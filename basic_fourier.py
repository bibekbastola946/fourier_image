import os
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from scipy.fft import fft2, fftshift, ifft2
from PIL import Image
from skimage.draw import disk


def load_image(url):
    """Load image from URL, resize, and convert to grayscale."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('L')
        return np.array(img, dtype=float)
    except Exception as e:
        print(f"Failed to load image from {url}: {e}")
        return None


def compute_fourier(image):
    """Compute Fourier Transform and extract magnitude and phase."""
    f_transform = fft2(image)
    f_shift = fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    phase_spectrum = np.angle(f_shift)
    return f_shift, magnitude_spectrum, phase_spectrum


def radial_profile(magnitude_spectrum):
    """Calculate the radial frequency profile."""
    rows, cols = magnitude_spectrum.shape
    center_row, center_col = rows // 2, cols // 2
    y, x = np.ogrid[-center_row:rows - center_row, -center_col:cols - center_col]
    radius = np.sqrt(x**2 + y**2)
    radius_int = radius.astype(int)

    radial_sum = np.bincount(radius_int.ravel(), weights=magnitude_spectrum.ravel())
    radial_count = np.bincount(radius_int.ravel())
    radial_profile = radial_sum / radial_count
    return radial_profile


def filter_frequencies(f_transform, mode="lowpass", radius=50):
    """Apply lowpass or highpass filtering."""
    rows, cols = f_transform.shape
    center = (rows // 2, cols // 2)

    mask = np.zeros((rows, cols), dtype=bool)
    rr, cc = disk(center, radius)
    if mode == "lowpass":
        mask[rr, cc] = True
    elif mode == "highpass":
        mask[rr, cc] = True
        mask = ~mask  # Invert the mask for highpass

    filtered_transform = f_transform * mask
    return filtered_transform


def visualize_fourier(image, magnitude_spectrum, phase_spectrum, radial_profile, image_id):
    """Visualize Fourier Transform results and filtered images."""
    os.makedirs("fourier_analysis", exist_ok=True)
    save_path = os.path.join("fourier_analysis", f"fourier_{image_id}.png")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Original Image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # Magnitude Spectrum (log scale)
    axes[0, 1].imshow(np.log1p(magnitude_spectrum), cmap='viridis')
    axes[0, 1].set_title("Magnitude Spectrum (log scale)")
    axes[0, 1].axis("off")

    # Phase Spectrum
    axes[0, 2].imshow(phase_spectrum, cmap='hsv')
    axes[0, 2].set_title("Phase Spectrum")
    axes[0, 2].axis("off")

    # Radial Frequency Profile
    axes[1, 0].plot(np.log1p(radial_profile))
    axes[1, 0].set_title("Radial Frequency Profile (Log Scale)")
    axes[1, 0].set_xlabel("Radius (pixels)")
    axes[1, 0].set_ylabel("Log(Average Power)")
    axes[1, 0].grid(True)

    # Filtered Images
    lowpass_transform = filter_frequencies(fftshift(fft2(image)), mode="lowpass", radius=30)
    highpass_transform = filter_frequencies(fftshift(fft2(image)), mode="highpass", radius=30)

    lowpass_image = np.abs(ifft2(fftshift(lowpass_transform)))
    highpass_image = np.abs(ifft2(fftshift(highpass_transform)))

    axes[1, 1].imshow(lowpass_image, cmap='gray')
    axes[1, 1].set_title("Lowpass Filtered Image")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(highpass_image, cmap='gray')
    axes[1, 2].set_title("Highpass Filtered Image")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Fourier analysis visualization for Image ID {image_id}.")


def analyze_metadata_unique(metadata_file, num_images=5):
    """Perform Fourier analysis for images with unique Equipment IDs."""
    # Load metadata and group by EquipmentId
    df = pd.read_csv(metadata_file)

    # Drop duplicates to ensure unique Equipment IDs
    df_unique = df.drop_duplicates(subset=['EquipmentId']).head(num_images)

    for idx, row in df_unique.iterrows():
        # Ensure unique filenames using EquipmentId and row index
        image_url = row['ImageURL']
        equipment_id = row.get('EquipmentId', f"Unknown_{idx}")
        image_id = f"{equipment_id}_{idx}"

        # Load image
        image = load_image(image_url)
        if image is None:
            print(f"Skipping Image ID {image_id} due to load failure.")
            continue

        # Compute Fourier Transform and related profiles
        f_transform, magnitude_spectrum, phase_spectrum = compute_fourier(image)
        radial_profile_data = radial_profile(magnitude_spectrum)

        # Visualize results
        visualize_fourier(image, magnitude_spectrum, phase_spectrum, radial_profile_data, image_id)


if __name__ == "__main__":
    metadata_file = "image_metadata.csv"  # Replace with your metadata file path
    analyze_metadata_unique(metadata_file, num_images=5)
