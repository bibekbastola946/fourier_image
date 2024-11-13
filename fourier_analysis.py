import numpy as np
import pandas as pd
from scipy.fft import fft2, fftshift, ifft2
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import signal


def load_and_process_image(url):
    """Load image from URL and convert to grayscale"""
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('L')
    return np.array(img, dtype=float)


def compute_enhanced_fourier_features(image):
    """
    Compute comprehensive Fourier transform features including frequency analysis,
    power spectrum, and various spectral properties
    """
    # Compute 2D FFT
    f_transform = fft2(image)
    f_shift = fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    power_spectrum = np.square(magnitude_spectrum)
    phase_spectrum = np.angle(f_shift)

    # Compute log spectrum for visualization
    log_spectrum = np.log1p(magnitude_spectrum)

    # Calculate radial frequency profile
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2
    y, x = np.ogrid[-center_row:rows - center_row, -center_col:cols - center_col]
    radius = np.sqrt(x * x + y * y)
    radius_int = radius.astype(int)

    # Radial profile of power spectrum
    radial_profile = np.bincount(radius_int.ravel(), power_spectrum.ravel())
    radial_profile = radial_profile / np.bincount(radius_int.ravel())

    # Frequency bands analysis
    total_power = np.sum(power_spectrum)
    low_freq_mask = radius < min(rows, cols) * 0.1
    mid_freq_mask = (radius >= min(rows, cols) * 0.1) & (radius < min(rows, cols) * 0.5)
    high_freq_mask = radius >= min(rows, cols) * 0.5

    # Directional analysis
    angles = np.rad2deg(np.arctan2(y, x))
    angles = np.where(angles < 0, angles + 360, angles)
    angle_bins = np.arange(0, 370, 10)
    direction_profile = np.histogram(angles, bins=angle_bins, weights=power_spectrum)[0]

    features = {
        # Basic spectral features
        'mean_magnitude': np.mean(magnitude_spectrum),
        'std_magnitude': np.std(magnitude_spectrum),
        'max_magnitude': np.max(magnitude_spectrum),
        'total_power': total_power,

        # Frequency band energies
        'low_freq_power': np.sum(power_spectrum * low_freq_mask) / total_power,
        'mid_freq_power': np.sum(power_spectrum * mid_freq_mask) / total_power,
        'high_freq_power': np.sum(power_spectrum * high_freq_mask) / total_power,

        # Phase statistics
        'phase_mean': np.mean(phase_spectrum),
        'phase_std': np.std(phase_spectrum),

        # Spectral moments
        'spectral_centroid': np.sum(radius * power_spectrum) / total_power,
        'spectral_spread': np.sqrt(
            np.sum(np.square(radius - np.sum(radius * power_spectrum) / total_power) * power_spectrum) / total_power),

        # Directional features
        'directional_mean': np.average(np.arange(len(direction_profile)), weights=direction_profile),
        'directional_std': np.sqrt(np.average((np.arange(len(direction_profile)) - np.average(
            np.arange(len(direction_profile)), weights=direction_profile)) ** 2, weights=direction_profile)),

        # Texture features
        'entropy': -np.sum(power_spectrum * np.log2(power_spectrum + 1e-10)) / np.log2(power_spectrum.size),
        'contrast': np.sum(radius * power_spectrum) / total_power
    }

    return features, {
        'magnitude_spectrum': magnitude_spectrum,
        'power_spectrum': power_spectrum,
        'phase_spectrum': phase_spectrum,
        'log_spectrum': log_spectrum,
        'radial_profile': radial_profile,
        'direction_profile': direction_profile,
        'angle_bins': angle_bins[:-1]
    }


def visualize_spectral_analysis(image, spectra, save_prefix='spectral'):
    """Create comprehensive visualizations of the spectral analysis"""
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4)

    # Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Magnitude spectrum
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(np.log1p(spectra['magnitude_spectrum']), cmap='viridis')
    ax2.set_title('Magnitude Spectrum (log scale)')
    ax2.axis('off')

    # Phase spectrum
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(spectra['phase_spectrum'], cmap='hsv')
    ax3.set_title('Phase Spectrum')
    ax3.axis('off')

    # Power spectrum
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(np.log1p(spectra['power_spectrum']), cmap='magma')
    ax4.set_title('Power Spectrum (log scale)')
    ax4.axis('off')

    # Radial profile
    ax5 = fig.add_subplot(gs[1, 0:2])
    ax5.plot(spectra['radial_profile'])
    ax5.set_title('Radial Frequency Profile')
    ax5.set_xlabel('Radius (pixels)')
    ax5.set_ylabel('Average Power')
    ax5.grid(True)

    # Directional profile
    ax6 = fig.add_subplot(gs[1, 2:], projection='polar')
    ax6.plot(np.deg2rad(spectra['angle_bins']), spectra['direction_profile'])
    ax6.set_title('Directional Power Distribution')

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_analysis.png')
    plt.close()


def analyze_equipment_images(csv_path, max_images=100):
    """Analyze images with enhanced Fourier features"""
    df = pd.read_csv(csv_path).head(max_images)
    all_features = []
    equipment_ids = []
    wear_percentages = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        try:
            img = load_and_process_image(row['ImageURL'])
            features, spectra = compute_enhanced_fourier_features(img)

            # Save spectral analysis for first few images
            if idx < 5:
                visualize_spectral_analysis(img, spectra, f'spectral_analysis_{row["EquipmentId"]}')

            all_features.append(features)
            equipment_ids.append(row['EquipmentId'])
            wear_percentages.append(row['PercentWear'])

        except Exception as e:
            print(f"\nError processing image {row['EquipmentId']}: {str(e)}")

    results_df = pd.DataFrame(all_features)
    results_df['EquipmentId'] = equipment_ids
    results_df['PercentWear'] = wear_percentages

    # Save results
    results_df.to_csv('enhanced_fourier_analysis.csv', index=False)
    return results_df


if __name__ == "__main__":
    csv_path = 'image_metadata.csv'
    results = analyze_equipment_images(csv_path)

    print("\nAnalysis complete! New features include:")
    print("- Frequency band power distribution")
    print("- Phase statistics")
    print("- Spectral moments")
    print("- Directional analysis")
    print("- Texture features")
    print("\nGenerated visualizations for the first 5 images show:")
    print("- Original image")
    print("- Magnitude spectrum")
    print("- Phase spectrum")
    print("- Power spectrum")
    print("- Radial frequency profile")
    print("- Directional power distribution")