import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_data(df):
    """Generates and saves visualizations of the dataset."""
    # Correlation Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    print(df.corr())
    plt.savefig("correlation_heatmap.png")
    plt.show()

    # Dataset Visualizations
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    sns.histplot(df['price'], kde=True)
    plt.title('Price Distribution')
    plt.savefig("price_distribution.png")

    plt.subplot(2, 3, 2)
    sns.scatterplot(x='area', y='price', data=df)
    plt.title('Area vs. Price')
    plt.savefig("area_vs_price.png")

    plt.subplot(2, 3, 3)
    sns.boxplot(x='bedrooms', y='price', data=df)
    plt.title('Bedrooms vs. Price')
    plt.savefig("bedrooms_vs_price.png")

    plt.subplot(2, 3, 4)
    sns.boxplot(x='bathrooms', y='price', data=df)
    plt.title('Bathrooms vs. Price')
    plt.savefig("bathrooms_vs_price.png")

    plt.subplot(2, 3, 5)
    sns.boxplot(x='stories', y='price', data=df)
    plt.title('Stories vs. Price')
    plt.savefig("stories_vs_price.png")

    plt.subplot(2, 3, 6)
    sns.boxplot(x='parking', y='price', data=df)
    plt.title('Parking vs. Price')
    plt.savefig("parking_vs_price.png")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("housing.csv")
    visualize_data(df)