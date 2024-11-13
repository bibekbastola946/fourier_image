# First, install required packages:
# pip install pandas numpy scipy Pillow requests matplotlib tqdm

import pandas as pd
import numpy as np
from scipy.fft import fft2, fftshift
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_and_process_image(url):
    """
    Load image from URL and convert to grayscale using PIL
    """
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('L')  # 'L' mode means grayscale
    return np.array(img)


def compute_fourier_features(image):
    """
    Compute 2D Fourier transform and extract relevant features
    """
    # Compute 2D FFT
    f_transform = fft2(image)
    f_shift = fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)

    # Extract features from the spectrum
    # Log scale for better handling of magnitude variations
    log_spectrum = np.log1p(magnitude_spectrum)

    features = {
        'mean_magnitude': np.mean(magnitude_spectrum),
        'std_magnitude': np.std(magnitude_spectrum),
        'max_magnitude': np.max(magnitude_spectrum),
        'energy': np.sum(np.square(magnitude_spectrum)),
        'low_freq_energy': np.sum(np.square(magnitude_spectrum[:image.shape[0] // 4, :image.shape[1] // 4])),
        'high_freq_energy': np.sum(np.square(magnitude_spectrum[3 * image.shape[0] // 4:, 3 * image.shape[1] // 4:])),
        'freq_ratio': np.sum(np.square(magnitude_spectrum[:image.shape[0] // 4, :image.shape[1] // 4])) /
                      np.sum(np.square(magnitude_spectrum[3 * image.shape[0] // 4:, 3 * image.shape[1] // 4:]))
    }

    return features, log_spectrum


def analyze_equipment_images(csv_path, max_images=100):
    """
    Analyze first 100 images in the dataset and correlate with wear percentage
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Limit to first max_images
    df = df.head(max_images)

    # Initialize lists to store results
    all_features = []
    equipment_ids = []
    wear_percentages = []
    failed_images = []

    # Use tqdm for progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        try:
            # Load and process image
            img = load_and_process_image(row['ImageURL'])

            # Compute Fourier features
            features, _ = compute_fourier_features(img)

            # Store results
            all_features.append(features)
            equipment_ids.append(row['EquipmentId'])
            wear_percentages.append(row['PercentWear'])

        except Exception as e:
            print(f"\nError processing image for EquipmentId {row['EquipmentId']}: {str(e)}")
            failed_images.append(row['EquipmentId'])

    # Convert to DataFrame
    results_df = pd.DataFrame(all_features)
    results_df['EquipmentId'] = equipment_ids
    results_df['PercentWear'] = wear_percentages

    # Compute correlations
    correlations = results_df.corr()['PercentWear'].sort_values(ascending=False)

    # Print summary
    print(f"\nProcessed {len(results_df)} images successfully")
    if failed_images:
        print(f"Failed to process {len(failed_images)} images: {failed_images}")

    return results_df, correlations


def visualize_spectrum(image_url, save_path=None):
    """
    Create visualization of original image and its Fourier spectrum
    """
    # Load image
    img = load_and_process_image(image_url)

    # Compute spectrum
    _, log_spectrum = compute_fourier_features(img)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(img, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2.imshow(log_spectrum, cmap='viridis')
    ax2.set_title('Fourier Spectrum (Log Scale)')
    ax2.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.close()


# Example usage
if __name__ == "__main__":
    # First install required packages if not already installed
    try:
        import pip

        required_packages = ['pandas', 'numpy', 'scipy', 'Pillow', 'requests', 'matplotlib', 'tqdm']
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                print(f"Installing {package}...")
                pip.main(['install', package])
    except Exception as e:
        print(f"Error installing packages: {str(e)}")
        print("Please manually run: pip install pandas numpy scipy Pillow requests matplotlib tqdm")

    csv_path = 'image_metadata.csv'

    # Process first 100 images
    results_df, correlations = analyze_equipment_images(csv_path, max_images=100)

    print("\nCorrelations with Wear Percentage:")
    print(correlations)

    # Save results to CSV
    results_df.to_csv('fourier_analysis_results.csv', index=False)
    print("\nResults saved to 'fourier_analysis_results.csv'")

    # Visualize first image as example
    first_image_url = pd.read_csv(csv_path)['ImageURL'].iloc[0]
    visualize_spectrum(first_image_url, 'fourier_analysis_example.png')
    print("Example visualization saved as 'fourier_analysis_example.png'")