import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def analyze_enhanced_results(csv_path='enhanced_fourier_analysis.csv'):
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Ensure PercentWear is numeric
    df['PercentWear'] = pd.to_numeric(df['PercentWear'], errors='coerce')

    # 1. Correlation Analysis
    spectral_features = ['mean_magnitude', 'std_magnitude', 'max_magnitude', 'total_power']
    freq_band_features = ['low_freq_power', 'high_freq_power']

    # Correlation of spectral features with PercentWear
    plt.figure(figsize=(10, 6))
    correlations = df[spectral_features + ['PercentWear']].corr()['PercentWear'].drop('PercentWear')
    sns.barplot(x=correlations.index, y=correlations.values)
    plt.title('Spectral Features - Correlation with Wear')
    plt.xticks(rotation=45)
    plt.ylabel('Correlation with PercentWear')
    plt.tight_layout()
    plt.savefig('correlation_spectral_features.png')
    plt.close()

    # Correlation of frequency bands with PercentWear
    plt.figure(figsize=(10, 6))
    correlations = df[freq_band_features + ['PercentWear']].corr()['PercentWear'].drop('PercentWear')
    sns.barplot(x=correlations.index, y=correlations.values)
    plt.title('Frequency Band Features - Correlation with Wear')
    plt.xticks(rotation=45)
    plt.ylabel('Correlation with PercentWear')
    plt.tight_layout()
    plt.savefig('correlation_frequency_bands.png')
    plt.close()

    # 2. Global Feature Importance
    all_correlations = df.corr()['PercentWear'].sort_values(ascending=False)
    print("\nTop 5 Most Predictive Features:")
    print(all_correlations[1:6])  # Exclude PercentWear itself

    # 3. Frequency Band Distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[freq_band_features])
    plt.title('Distribution of Power Across Frequency Bands')
    plt.tight_layout()
    plt.savefig('frequency_bands_distribution.png')
    plt.close()

    # 4. Scatter Plots for Key Relationships
    # Scatter plot of mean_magnitude vs PercentWear
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='mean_magnitude', y='PercentWear', size='total_power', hue='PercentWear', palette='viridis')
    plt.title('Mean Magnitude vs PercentWear')
    plt.xlabel('Mean Magnitude')
    plt.ylabel('PercentWear')
    plt.tight_layout()
    plt.savefig('mean_magnitude_vs_percentwear.png')
    plt.close()

    # Scatter plot of low_freq_power vs PercentWear
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='low_freq_power', y='PercentWear', size='total_power', hue='PercentWear', palette='viridis')
    plt.title('Low Frequency Power vs PercentWear')
    plt.xlabel('Low Frequency Power')
    plt.ylabel('PercentWear')
    plt.tight_layout()
    plt.savefig('low_freq_power_vs_percentwear.png')
    plt.close()

    # 5. Total Power Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['total_power'], bins=30, kde=True)
    plt.title('Distribution of Total Power')
    plt.xlabel('Total Power')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('total_power_distribution.png')
    plt.close()

    print("\nAnalysis complete! Generated visualizations for:")
    print("- Spectral and frequency band correlations")
    print("- Frequency band distribution")
    print("- Scatter plots for key relationships")
    print("- Total power distribution")

    return df


if __name__ == "__main__":
    try:
        results = analyze_enhanced_results()
    except Exception as e:
        print(f"Error during analysis: {e}")
