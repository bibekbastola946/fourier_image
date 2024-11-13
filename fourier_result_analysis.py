import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np


def analyze_enhanced_results(csv_path='enhanced_fourier_analysis.csv'):
    df = pd.read_csv(csv_path)

    # 1. Feature groups analysis
    feature_groups = {
        'Basic Spectral': ['mean_magnitude', 'std_magnitude', 'max_magnitude', 'total_power'],
        'Frequency Bands': ['low_freq_power', 'mid_freq_power', 'high_freq_power'],
        'Phase': ['phase_mean', 'phase_std'],
        'Spectral Shape': ['spectral_centroid', 'spectral_spread'],
        'Directional': ['directional_mean', 'directional_std'],
        'Texture': ['entropy', 'contrast']
    }

    # Create correlation plots for each feature group
    for group_name, features in feature_groups.items():
        plt.figure(figsize=(10, 6))

        # Get correlations with wear percentage
        correlations = df[features + ['PercentWear']].corr()['PercentWear'].drop('PercentWear')

        # Create bar plot
        sns.barplot(x=correlations.index, y=correlations.values)
        plt.title(f'{group_name} Features - Correlation with Wear')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'correlation_{group_name.lower().replace(" ", "_")}.png')
        plt.close()

    # 2. Feature importance analysis
    all_correlations = df.corr()['PercentWear'].sort_values(ascending=False)

    print("Top 5 Most Predictive Features:")
    print(all_correlations[1:6])  # Skip PercentWear itself

    # 3. Frequency band analysis
    plt.figure(figsize=(10, 6))
    freq_bands = ['low_freq_power', 'mid_freq_power', 'high_freq_power']
    sns.boxplot(data=df[freq_bands])
    plt.title('Distribution of Power Across Frequency Bands')
    plt.tight_layout()
    plt.savefig('frequency_bands_distribution.png')
    plt.close()

    # 4. Phase analysis
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='phase_mean', y='phase_std', hue='PercentWear', size='total_power')
    plt.title('Phase Characteristics vs Wear')
    plt.tight_layout()
    plt.savefig('phase_analysis.png')
    plt.close()

    return df


if __name__ == "__main__":
    results = analyze_enhanced_results()
    print("\nAnalysis complete! Generated visualizations for:")
    print("- Correlation plots for each feature group")
    print("- Frequency band distribution")
    print("- Phase characteristics")