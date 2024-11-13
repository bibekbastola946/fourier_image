import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np


def analyze_fourier_results(csv_path='fourier_analysis_results.csv'):
    # Read the results
    df = pd.read_csv(csv_path)

    # 1. Basic Statistical Summary
    print("Statistical Summary of Features:")
    print(df.describe())

    # 2. Correlation Analysis with Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Fourier Features')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()

    # 3. Scatter plots of top correlated features with PercentWear
    correlations = df.corr()['PercentWear'].sort_values(ascending=False)
    top_features = correlations[1:4].index  # Skip PercentWear itself

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, feature in enumerate(top_features):
        sns.scatterplot(data=df, x=feature, y='PercentWear', ax=axes[i])
        axes[i].set_title(f'{feature} vs PercentWear')

        # Add trend line
        z = np.polyfit(df[feature], df['PercentWear'], 1)
        p = np.poly1d(z)
        axes[i].plot(df[feature], p(df[feature]), "r--", alpha=0.8)

        # Calculate and display R-squared
        r2 = stats.pearsonr(df[feature], df['PercentWear'])[0] ** 2
        axes[i].text(0.05, 0.95, f'R² = {r2:.3f}',
                     transform=axes[i].transAxes,
                     bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('top_correlations.png')
    plt.close()

    # 4. Feature importance analysis
    print("\nFeature Correlations with Wear Percentage:")
    for feature, corr in correlations.items():
        print(f"{feature}: {corr:.3f}")

    # 5. Wear percentage distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='PercentWear', bins=20)
    plt.title('Distribution of Wear Percentages')
    plt.xlabel('Wear Percentage')
    plt.ylabel('Count')
    plt.savefig('wear_distribution.png')
    plt.close()

    # 6. Identify potential outliers
    print("\nPotential Outliers (> 2 standard deviations from mean):")
    for column in df.select_dtypes(include=[np.number]).columns:
        mean = df[column].mean()
        std = df[column].std()
        outliers = df[abs(df[column] - mean) > 2 * std]
        if len(outliers) > 0:
            print(f"\n{column}:")
            print(f"Number of outliers: {len(outliers)}")
            print(f"Outlier values: {outliers[column].values}")

    return df


if __name__ == "__main__":
    # Analyze the results
    df = analyze_fourier_results()

    print("\nAnalysis complete! Generated visualizations:")
    print("1. correlation_heatmap.png - Shows correlations between all features")
    print("2. top_correlations.png - Scatter plots of top correlated features with wear percentage")
    print("3. wear_distribution.png - Distribution of wear percentages")