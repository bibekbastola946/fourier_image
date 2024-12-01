import os
import numpy as np
import pandas as pd
from scipy.fft import fft2, fftshift
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def load_image(url, resize_to=(256, 256)):
    """Load image from URL, resize, and convert to grayscale."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('L')
        img = img.resize(resize_to, Image.Resampling.LANCZOS)
        return np.array(img, dtype=float)
    except Exception as e:
        print(f"Failed to load image from {url}: {e}")
        return None


def compute_fft_features(image):
    """Compute essential Fourier transform features."""
    # Compute 2D FFT
    f_transform = fft2(image)
    f_shift = fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    power_spectrum = np.square(magnitude_spectrum)

    # Radial frequency profile
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2
    y, x = np.ogrid[-center_row:rows - center_row, -center_col:cols - center_col]
    radius = np.sqrt(x**2 + y**2)
    radius_int = radius.astype(int)

    radial_profile = np.bincount(radius_int.ravel(), power_spectrum.ravel())
    radial_profile /= np.bincount(radius_int.ravel())

    # Frequency bands analysis
    total_power = np.sum(power_spectrum)
    low_freq_mask = radius < min(rows, cols) * 0.1
    high_freq_mask = radius >= min(rows, cols) * 0.5

    features = {
        'mean_magnitude': np.mean(magnitude_spectrum),
        'std_magnitude': np.std(magnitude_spectrum),
        'max_magnitude': np.max(magnitude_spectrum),
        'total_power': total_power,
        'low_freq_power': np.sum(power_spectrum[low_freq_mask]) / total_power,
        'high_freq_power': np.sum(power_spectrum[high_freq_mask]) / total_power,
    }

    return features, magnitude_spectrum, power_spectrum


def process_image(row):
    """Process a single image and compute FFT features."""
    image_url = row['ImageURL']
    equipment_id = row.get('EquipmentId', 'unknown')
    percent_wear = row.get('PercentWear', None)

    # Load image
    image = load_image(image_url)
    if image is None:
        return None

    # Compute FFT features
    features, magnitude_spectrum, power_spectrum = compute_fft_features(image)

    # Return results for later visualization
    features['EquipmentId'] = equipment_id
    features['PercentWear'] = percent_wear
    return features, image, magnitude_spectrum, power_spectrum


def visualize_images(results, visualize_count=2):
    """Visualize results on the main thread."""
    os.makedirs("spectral_analysis", exist_ok=True)

    for idx, (features, image, magnitude_spectrum, power_spectrum) in enumerate(results[:visualize_count]):
        save_path = os.path.join("spectral_analysis", f"spectral_{features['EquipmentId']}.png")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original Image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Magnitude Spectrum (log scale)
        axes[1].imshow(np.log1p(magnitude_spectrum), cmap='viridis')
        axes[1].set_title("Magnitude Spectrum (log scale)")
        axes[1].axis("off")

        # Power Spectrum (log scale)
        axes[2].imshow(np.log1p(power_spectrum), cmap='magma')
        axes[2].set_title("Power Spectrum (log scale)")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


def analyze_images_parallel(metadata_file, max_images=10000, visualize_count=2):
    """Analyze images with parallel processing and visualize on the main thread."""
    df = pd.read_csv(metadata_file).head(max_images)

    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for result in tqdm(executor.map(process_image, df.to_dict('records')), total=len(df), desc="Processing Images"):
            if result:
                results.append(result)

    # Extract features for saving to CSV
    features = [res[0] for res in results]
    results_df = pd.DataFrame(features)
    results_df.to_csv('enhanced_fourier_analysis.csv', index=False)

    # Perform visualization on the main thread
    visualize_images(results, visualize_count=visualize_count)

    print("\nAnalysis complete!")
    print(f"Results saved to 'enhanced_fourier_analysis.csv'.")
    print(f"Visualizations saved in the 'spectral_analysis' directory.")


if __name__ == "__main__":
    metadata_file = 'image_metadata.csv'
    if not os.path.exists(metadata_file):
        print(f"Metadata file not found: {metadata_file}")
    else:
        analyze_images_parallel(metadata_file, max_images=10000, visualize_count=2)
